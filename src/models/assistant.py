"""Assistant model: 3D ResNet-18 (slow_r18) from pytorchvideo.

Supports optional intermediate feature extraction for attention transfer.
"""

from typing import Optional

import torch
import torch.nn as nn


class AssistantModel(nn.Module):
    """Wrapper around pytorchvideo's slow_r18 with configurable head and
    optional intermediate feature extraction for attention transfer.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to load Kinetics-400 pretrained weights.
        freeze_backbone: If True, freeze all layers except the final head.
        extract_features: If True, store intermediate activations via hooks.
    """

    FEATURE_BLOCKS = [2, 3, 4]

    def __init__(
        self,
        num_classes: int = 101,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        extract_features: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.extract_features = extract_features
        self._intermediate_features: dict[int, torch.Tensor] = {}
        self._hooks: list = []

        print("[Assistant] Creating ResNet-18 model...")
        self._backend = "torchvision"
        from torchvision.models.video import r3d_18

        if pretrained:
            print("[Assistant] Loading pretrained Kinetics-400 weights for r3d_18...")
            import os
            import torch
            from pathlib import Path
            
            offline_path = Path("experiments/checkpoints/r3d_18-b3b3357e.pth")
            if offline_path.is_file():
                print(f"[Assistant] Found offline weights at: {offline_path}")
                try:
                    self.model = r3d_18(weights=None)
                except TypeError:
                    self.model = r3d_18(pretrained=False)
                state_dict = torch.load(str(offline_path), map_location="cpu")
                self.model.load_state_dict(state_dict)
                print("[Assistant] Offline weights loaded successfully.")
            else:
                print(f"[Assistant] WARNING: Offline weights not found at {offline_path}")
                print("[Assistant] Attempting to download via torchvision...")
                try:
                    self.model = r3d_18(weights="DEFAULT")
                except TypeError:
                    self.model = r3d_18(pretrained=True)
        else:
            try:
                self.model = r3d_18(weights=None)
            except TypeError:
                self.model = r3d_18(pretrained=False)

        print("[Assistant] torchvision r3d_18 ready.")

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            self._freeze_backbone()

        if extract_features:
            self._register_hooks()

    def _freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
        if self._backend == "pytorchvideo":
            for param in self.model.blocks[self._head_idx].proj.parameters():
                param.requires_grad = True
        else:
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def _register_hooks(self) -> None:
        self._clear_hooks()
        if self._backend == "pytorchvideo":
            for block_idx in self.FEATURE_BLOCKS:
                hook = self.model.blocks[block_idx].register_forward_hook(
                    self._make_hook(block_idx)
                )
                self._hooks.append(hook)
        else:
            layer_map = {
                2: self.model.layer2,
                3: self.model.layer3,
                4: self.model.layer4,
            }
            for block_idx, layer in layer_map.items():
                hook = layer.register_forward_hook(self._make_hook(block_idx))
                self._hooks.append(hook)

    def _make_hook(self, block_idx: int):
        def hook_fn(module, input, output):
            self._intermediate_features[block_idx] = output
        return hook_fn

    def _clear_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_intermediate_features(self) -> dict[int, torch.Tensor]:
        return self._intermediate_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._intermediate_features.clear()
        return self.model(x)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if self._backend == "pytorchvideo":
            for i, block in enumerate(self.model.blocks):
                if i < self._head_idx:
                    x = block(x)
                elif i == self._head_idx:
                    if hasattr(block, "pool"):
                        x = block.pool(x)
                    if hasattr(block, "dropout"):
                        x = block.dropout(x)
                    x = x.flatten(1)
                    break
            return x

        x = self.model.stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.flatten(1)
        return x


def get_assistant(
    num_classes: int = 101,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    extract_features: bool = False,
    checkpoint_path: Optional[str] = None,
) -> AssistantModel:
    """Factory function to create and optionally load an assistant model."""
    model = AssistantModel(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        extract_features=extract_features,
    )

    if checkpoint_path is not None:
        print(f"[Assistant] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print("[Assistant] Checkpoint loaded.")

    return model
