"""Loss functions for Knowledge Distillation and Attention Transfer."""

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


def _spatial_attention_map(features: torch.Tensor) -> torch.Tensor:
    """Compute spatial attention map from a feature tensor.

    A(F) = normalize(sum(|F_c|^2, dim=channel))

    Args:
        features: Feature tensor [B, C, T, H, W] or [B, C, H, W].

    Returns:
        Normalized attention map [B, T*H*W] or [B, H*W].
    """
    # Sum of squared activations across channel dimension
    attn = (features ** 2).sum(dim=1)  # [B, T, H, W] or [B, H, W]
    # Flatten spatial dimensions
    attn = attn.view(attn.size(0), -1)  # [B, N]
    # L2-normalize
    attn = F.normalize(attn, p=2, dim=1)
    return attn


class AttentionTransferLoss(nn.Module):
    """Attention Transfer loss between teacher and student intermediate features.

    Computes MSE between L2-normalized spatial attention maps at matching layers.
        L_AT = β * Σ_i MSE(A_teacher_i, A_student_i)

    Args:
        beta: Weight for the attention transfer loss.
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        teacher_features: dict[int, torch.Tensor],
        student_features: dict[int, torch.Tensor],
        teacher_keys: list[int],
        student_keys: list[int],
    ) -> torch.Tensor:
        """Compute AT loss over paired feature maps.

        Args:
            teacher_features: Dict of {block_idx: feature_tensor} from teacher.
            student_features: Dict of {stage_idx: feature_tensor} from student.
            teacher_keys: Ordered list of teacher block indices to use.
            student_keys: Ordered list of student stage indices to use (same length).

        Returns:
            Scalar AT loss.
        """
        assert len(teacher_keys) == len(student_keys), \
            "Teacher and student must have matching number of AT layers."

        at_loss = torch.tensor(0.0, device=next(iter(teacher_features.values())).device)

        for t_key, s_key in zip(teacher_keys, student_keys):
            t_attn = _spatial_attention_map(teacher_features[t_key])
            s_attn = _spatial_attention_map(student_features[s_key])

            # If spatial sizes differ, the attention maps are already flattened + normalized.
            # MSE over the normalized attention vectors.
            # We need to handle size mismatch: interpolate the smaller to match the larger.
            if t_attn.shape != s_attn.shape:
                # Resize student attention to match teacher
                s_attn = F.interpolate(
                    s_attn.unsqueeze(1), size=t_attn.shape[1], mode="linear", align_corners=False
                ).squeeze(1)
                s_attn = F.normalize(s_attn, p=2, dim=1)

            at_loss = at_loss + F.mse_loss(s_attn, t_attn)

        return self.beta * at_loss


class CombinedKDATLoss(nn.Module):
    """Combined Knowledge Distillation + Attention Transfer loss.

    Args:
        temperature: KD softmax temperature.
        alpha: KD weight (distillation vs CE).
        beta: AT loss weight.
        teacher_keys: Teacher block indices for AT.
        student_keys: Student stage indices for AT.
    """

    def __init__(
        self,
        temperature: float = 5.0,
        alpha: float = 0.7,
        beta: float = 0.1,
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
        self.at_loss = AttentionTransferLoss(beta=beta)
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
        at = self.at_loss(teacher_features, student_features, self.teacher_keys, self.student_keys)
        total = kd + at
        return total, kd, at
