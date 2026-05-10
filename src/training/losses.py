"""Loss functions for Knowledge Distillation and Attention Transfer.

Attention Transfer (AT) decomposes 3D video features [B, C, T, H, W] into:
  - Spatial attention [B, H, W]: where to look (aggregated over channels and time)
  - Temporal attention [B, T]: when activations peak (aggregated over channels and space)

This separation is critical for video: spatial and temporal dimensions carry different
semantic information (location vs. motion dynamics), and mixing them into a single
flattened vector (as in the original 2D AT paper extension) loses this distinction.

References:
  - Zagoruyko & Komodakis, "Paying More Attention to Attention", ICLR 2017
  - Extra objective Track 6: "Attention Transfer across intermediate temporal
    activation mappings"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """Standard Knowledge Distillation loss.

    Combines soft-target KL divergence with hard-target cross-entropy:
        L = α * T² * KL(student_soft || teacher_soft) + (1 - α) * CE(student, labels)

    Args:
        temperature: Softmax temperature for smoothing logits.
        alpha: Weight for the distillation loss (1-alpha for CE).
    """

    def __init__(self, temperature: float = 5.0, alpha: float = 0.7, label_smoothing: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KD loss.

        Args:
            student_logits: Raw logits from student [B, C].
            teacher_logits: Raw logits from teacher [B, C].
            labels: Ground truth class indices [B].

        Returns:
            Combined scalar loss.
        """
        T = self.temperature

        # Soft targets
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL divergence (batchmean reduction) scaled by T²
        kd_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        # Hard-target cross entropy
        ce_loss = self.ce_loss(student_logits, labels)

        return self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss


# ---------------------------------------------------------------------------
# Attention map functions for 3D video features
# ---------------------------------------------------------------------------

def _spatial_attention_map(features: torch.Tensor) -> torch.Tensor:
    """Compute spatial attention map from 3D video features.

    Aggregates over channels and time to produce a per-location activation
    intensity: "where does the network look across all frames?"

    A_spatial(F) = L2_normalize( mean(F², dim=[C, T]) )

    For 5D input [B, C, T, H, W]:
        mean over C and T → [B, H, W] → flatten → [B, H*W] → L2-normalize

    For 4D input [B, C, H, W] (2D fallback):
        mean over C → [B, H, W] → flatten → [B, H*W] → L2-normalize

    Args:
        features: Feature tensor [B, C, T, H, W] or [B, C, H, W].

    Returns:
        Normalized spatial attention map [B, H*W].
    """
    features = features.float()
    if features.dim() == 5:
        # 3D: aggregate over channels (1) and time (2)
        attn = (features ** 2).mean(dim=(1, 2))  # [B, H, W]
    elif features.dim() == 4:
        # 2D fallback: aggregate over channels (1)
        attn = (features ** 2).mean(dim=1)  # [B, H, W]
    else:
        raise ValueError(f"Expected 4D or 5D features, got {features.dim()}D")

    attn = attn.view(attn.size(0), -1)  # [B, H*W]
    attn = F.normalize(attn, p=2, dim=1)
    return attn


def _temporal_attention_map(features: torch.Tensor) -> torch.Tensor:
    """Compute temporal attention map from 3D video features.

    Aggregates over channels and spatial dimensions to produce a per-timestep
    activation intensity: "when does the network activate most?"

    A_temporal(F) = L2_normalize( mean(F², dim=[C, H, W]) )

    [B, C, T, H, W] → mean over C, H, W → [B, T] → L2-normalize

    Args:
        features: Feature tensor [B, C, T, H, W].

    Returns:
        Normalized temporal attention map [B, T].
    """
    features = features.float()
    if features.dim() != 5:
        raise ValueError(
            f"Temporal attention requires 5D features [B, C, T, H, W], got {features.dim()}D"
        )

    # Aggregate over channels (1), height (3), width (4)
    attn = (features ** 2).mean(dim=(1, 3, 4))  # [B, T]
    attn = F.normalize(attn, p=2, dim=1)
    return attn


class AttentionTransferLoss(nn.Module):
    """Attention Transfer loss between teacher and student intermediate features.

    Computes separate spatial and temporal AT losses:
        L_AT = β_s * Σ_i MSE(A_spatial_t_i, A_spatial_s_i)
             + β_t * Σ_i MSE(A_temporal_t_i, A_temporal_s_i)

    Spatial attention maps may have different H×W between teacher and student;
    this is handled via 2D bilinear interpolation (correct for spatial grids).

    Temporal attention maps have the same T in both models (both use temporal
    stride 1), so no interpolation is needed — the signal is transferred directly.

    Args:
        beta_spatial: Weight for spatial AT loss component.
        beta_temporal: Weight for temporal AT loss component.
    """

    def __init__(self, beta_spatial: float = 0.05, beta_temporal: float = 0.05):
        super().__init__()
        self.beta_spatial = beta_spatial
        self.beta_temporal = beta_temporal

    def forward(
        self,
        teacher_features: dict[int, torch.Tensor],
        student_features: dict[int, torch.Tensor],
        teacher_keys: list[int],
        student_keys: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute AT loss over paired feature maps.

        Args:
            teacher_features: Dict of {block_idx: feature_tensor} from teacher.
            student_features: Dict of {stage_idx: feature_tensor} from student.
            teacher_keys: Ordered list of teacher block indices to use.
            student_keys: Ordered list of student stage indices to use (same length).

        Returns:
            Tuple of (total_at_loss, spatial_at_loss, temporal_at_loss).
        """
        assert len(teacher_keys) == len(student_keys), \
            "Teacher and student must have matching number of AT layers."

        device = next(iter(teacher_features.values())).device
        spatial_loss = torch.tensor(0.0, device=device)
        temporal_loss = torch.tensor(0.0, device=device)

        for t_key, s_key in zip(teacher_keys, student_keys):
            t_feat = teacher_features[t_key]
            s_feat = student_features[s_key]

            # --- Spatial AT ---
            if self.beta_spatial > 0:
                t_spatial = _spatial_attention_map(t_feat)  # [B, H_t * W_t]
                s_spatial = _spatial_attention_map(s_feat)  # [B, H_s * W_s]

                if t_spatial.shape != s_spatial.shape:
                    # Proper 2D interpolation: reshape to [B, 1, H, W], interpolate, re-flatten
                    _, _, _, H_t, W_t = t_feat.shape
                    _, _, _, H_s, W_s = s_feat.shape
                    s_spatial_2d = s_spatial.view(-1, 1, H_s, W_s)
                    s_spatial_2d = F.interpolate(
                        s_spatial_2d, size=(H_t, W_t), mode="bilinear", align_corners=False
                    )
                    s_spatial = s_spatial_2d.view(s_spatial_2d.size(0), -1)  # [B, H_t*W_t]
                    s_spatial = F.normalize(s_spatial, p=2, dim=1)

                spatial_loss = spatial_loss + F.mse_loss(s_spatial, t_spatial)

            # --- Temporal AT ---
            if self.beta_temporal > 0 and t_feat.dim() == 5:
                t_temporal = _temporal_attention_map(t_feat)  # [B, T]
                s_temporal = _temporal_attention_map(s_feat)  # [B, T]

                # T should match (both models have temporal stride 1), but handle
                # edge cases defensively
                if t_temporal.shape != s_temporal.shape:
                    s_temporal = F.interpolate(
                        s_temporal.unsqueeze(1), size=t_temporal.shape[1],
                        mode="linear", align_corners=False,
                    ).squeeze(1)
                    s_temporal = F.normalize(s_temporal, p=2, dim=1)

                temporal_loss = temporal_loss + F.mse_loss(s_temporal, t_temporal)

        total = self.beta_spatial * spatial_loss + self.beta_temporal * temporal_loss
        return total, self.beta_spatial * spatial_loss, self.beta_temporal * temporal_loss


class CombinedKDATLoss(nn.Module):
    """Combined Knowledge Distillation + Attention Transfer loss.

    Total loss = L_KD + L_AT_spatial + L_AT_temporal

    Where:
        L_KD = α * T² * KL(student_soft || teacher_soft) + (1-α) * CE
        L_AT_spatial = β_s * Σ_i MSE(A_spatial_teacher_i, A_spatial_student_i)
        L_AT_temporal = β_t * Σ_i MSE(A_temporal_teacher_i, A_temporal_student_i)

    Args:
        temperature: KD softmax temperature.
        alpha: KD weight (distillation vs CE).
        beta_spatial: Spatial AT loss weight.
        beta_temporal: Temporal AT loss weight.
        teacher_keys: Teacher block indices for AT.
        student_keys: Student stage indices for AT.
    """

    def __init__(
        self,
        temperature: float = 5.0,
        alpha: float = 0.7,
        beta_spatial: float = 0.05,
        beta_temporal: float = 0.05,
        label_smoothing: float = 0.0,
        teacher_keys: list[int] | None = None,
        student_keys: list[int] | None = None,
    ):
        super().__init__()
        self.kd_loss = KDLoss(
            temperature=temperature,
            alpha=alpha,
            label_smoothing=label_smoothing,
        )
        self.at_loss = AttentionTransferLoss(
            beta_spatial=beta_spatial,
            beta_temporal=beta_temporal,
        )
        self.teacher_keys = teacher_keys or [3, 4, 5]
        self.student_keys = student_keys or [2, 4, 6]

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_features: dict[int, torch.Tensor],
        student_features: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute combined loss.

        Returns:
            Tuple of (total_loss, kd_component, at_component).
        """
        kd = self.kd_loss(student_logits, teacher_logits, labels)
        at_total, at_spatial, at_temporal = self.at_loss(
            teacher_features, student_features, self.teacher_keys, self.student_keys,
        )
        total = kd + at_total
        return total, kd, at_total
