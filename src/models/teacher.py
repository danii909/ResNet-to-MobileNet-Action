"""Teacher model: 3D ResNet-50 (slow_r50) from pytorchvideo.

Pretrained on Kinetics-400, adapted for UCF-101 (101 classes).
Supports intermediate feature extraction for Attention Transfer.
"""

from typing import Optional

import torch
import torch.nn as nn


class TeacherModel(nn.Module):
    """Wrapper around pytorchvideo's slow_r50 with configurable head and
    optional intermediate feature extraction for attention transfer.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load Kinetics-400 pretrained weights.
        freeze_backbone: If True, freeze all layers except the final head.
        extract_features: If True, store intermediate activations via hooks.
    """

    # Indices of residual stages to hook for attention transfer.
    # Block 5 is the classification head (outputs [B, num_classes], 2D) — excluded.
    FEATURE_BLOCKS = [2, 3, 4]

    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        extract_features: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.extract_features = extract_features
        self._intermediate_features: dict[int, torch.Tensor] = {}
        self._hooks: list = []

        # Load slow_r50 from pytorchvideo (local import, no torch.hub)
        print("[Teacher] Creating slow_r50 model...")
        from pytorchvideo.models.hub import slow_r50
        self.model = slow_r50(pretrained=False)

        if pretrained:
            print("[Teacher] Loading pretrained weights from local cache...")
            import os
            from pathlib import Path

            # Check common cache locations
            cache_paths = [
                Path("experiments/checkpoints/SLOW_8x8_R50.pyth"),
                Path.home() / "dl26-projects" / "experiments" / "checkpoints" / "SLOW_8x8_R50.pyth",
                Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "SLOW_8x8_R50.pyth",
            ]
            env_weights = os.environ.get("SLOW_R50_WEIGHTS", "")
            if env_weights:
                cache_paths.insert(0, Path(env_weights))
            weights_path = None
            for p in cache_paths:
                if p.is_file():
                    weights_path = p
                    break

            if weights_path is not None:
                print(f"[Teacher] Found weights at: {weights_path}")
                state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
                if "model_state" in state_dict:
                    state_dict = state_dict["model_state"]
                self.model.load_state_dict(state_dict, strict=False)
                print("[Teacher] Pretrained weights loaded.")
            else:
                print("[Teacher] WARNING: Pretrained weights not found!")
                print("[Teacher]   Expected at: experiments/checkpoints/SLOW_8x8_R50.pyth")
                print("[Teacher]   Upload them with: .\\sync_cluster.ps1 -Action upload")
                print("[Teacher] Continuing with random initialization.")

        print("[Teacher] slow_r50 ready.")

        # Find the head block dynamically (last block with a .proj linear layer)
        head_idx = None
        for i in range(len(self.model.blocks) - 1, -1, -1):
            if hasattr(self.model.blocks[i], "proj") and isinstance(
                self.model.blocks[i].proj, nn.Linear
            ):
                head_idx = i
                break
        if head_idx is None:
            raise RuntimeError(
                f"[Teacher] Cannot find head block with .proj in model.blocks "
                f"(len={len(self.model.blocks)}). "
                f"Blocks: {[type(b).__name__ for b in self.model.blocks]}"
            )
        self._head_idx = head_idx
        print(f"[Teacher] Head block found at index {head_idx} "
              f"(total blocks: {len(self.model.blocks)})")

        in_features = self.model.blocks[head_idx].proj.in_features
        self.model.blocks[head_idx].proj = nn.Linear(in_features, num_classes)

        # Replace fixed AvgPool3d with adaptive pooling to support any input size
        if hasattr(self.model.blocks[head_idx], "pool"):
            self.model.blocks[head_idx].pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if freeze_backbone:
            self._freeze_backbone()

        if extract_features:
            self._register_hooks()

    def _freeze_backbone(self) -> None:
        """Freeze all parameters except the classification head."""
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.blocks[self._head_idx].proj.parameters():
            param.requires_grad = True

    def _register_hooks(self) -> None:
        """Register forward hooks on residual blocks to capture intermediate features."""
        self._clear_hooks()
        for block_idx in self.FEATURE_BLOCKS:
            hook = self.model.blocks[block_idx].register_forward_hook(
                self._make_hook(block_idx)
            )
            self._hooks.append(hook)

    def _make_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            self._intermediate_features[block_idx] = output
        return hook_fn

    def _clear_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def get_intermediate_features(self) -> dict[int, torch.Tensor]:
        """Return intermediate feature maps captured during the last forward pass."""
        return self._intermediate_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, T, H, W].

        Returns:
            Logits of shape [B, num_classes].
        """
        self._intermediate_features.clear()
        return self.model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pre-classifier embedding (before final linear layer).

        Returns:
            Embedding tensor of shape [B, feature_dim].
        """
        # Forward through all blocks except the head's projection
        for i, block in enumerate(self.model.blocks):
            if i < 6:
                x = block(x)
            elif i == 6:
                # Head block: pool + dropout but skip proj
                # blocks[6] is ResNetBasicHead: pool -> dropout -> proj -> activation
                x = block.pool(x)
                x = block.dropout(x)
                x = x.flatten(1)
                break
        return x


def get_teacher(
    num_classes: int = 101,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    extract_features: bool = False,
    checkpoint_path: Optional[str] = None,
) -> TeacherModel:
    """Factory function to create and optionally load a teacher model.

    Args:
        num_classes: Number of output classes.
        pretrained: Load Kinetics-400 pretrained weights.
        freeze_backbone: Freeze backbone, train only head.
        extract_features: Enable intermediate feature hooks.
        checkpoint_path: Optional path to a fine-tuned checkpoint.

    Returns:
        Configured TeacherModel instance.
    """
    model = TeacherModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        extract_features=extract_features,
    )

    if checkpoint_path is not None:
        print(f"[Teacher] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print("[Teacher] Checkpoint loaded.")

    return model
