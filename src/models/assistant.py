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
        slow_r18 = None
        try:
            from pytorchvideo.models.hub import slow_r18 as _slow_r18
            slow_r18 = _slow_r18
        except ImportError:
            slow_r18 = None

        if slow_r18 is not None:
            self._backend = "pytorchvideo"
            self.model = slow_r18(pretrained=False)

            if pretrained:
                print("[Assistant] Loading pretrained weights from local cache...")
                import os
                from pathlib import Path

                cache_paths = [
                    Path("experiments/checkpoints/SLOW_8x8_R18.pyth"),
                    Path.home() / "dl26-projects" / "experiments" / "checkpoints" / "SLOW_8x8_R18.pyth",
                    Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "SLOW_8x8_R18.pyth",
                ]
                env_weights = os.environ.get("SLOW_R18_WEIGHTS", "")
                if env_weights:
                    cache_paths.insert(0, Path(env_weights))

                weights_path = None
                for path in cache_paths:
                    if path.is_file():
                        weights_path = path
                        break

                if weights_path is not None:
                    print(f"[Assistant] Found weights at: {weights_path}")
                    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
                    if "model_state" in state_dict:
                        state_dict = state_dict["model_state"]
                    self.model.load_state_dict(state_dict, strict=False)
                    print("[Assistant] Pretrained weights loaded.")
                else:
                    print("[Assistant] WARNING: Pretrained weights not found!")
                    print("[Assistant]   Expected at: experiments/checkpoints/SLOW_8x8_R18.pyth")
                    print("[Assistant] Continuing with random initialization.")

            print("[Assistant] slow_r18 ready.")

            head_idx = None
            for i in range(len(self.model.blocks) - 1, -1, -1):
                if hasattr(self.model.blocks[i], "proj") and isinstance(
                    self.model.blocks[i].proj, nn.Linear
                ):
                    head_idx = i
                    break
            if head_idx is None:
                raise RuntimeError(
                    f"[Assistant] Cannot find head block with .proj in model.blocks "
                    f"(len={len(self.model.blocks)}). "
                    f"Blocks: {[type(b).__name__ for b in self.model.blocks]}"
                )
            self._head_idx = head_idx
            print(f"[Assistant] Head block found at index {head_idx} "
                  f"(total blocks: {len(self.model.blocks)})")

            in_features = self.model.blocks[head_idx].proj.in_features
            self.model.blocks[head_idx].proj = nn.Linear(in_features, num_classes)

            if hasattr(self.model.blocks[head_idx], "pool"):
                self.model.blocks[head_idx].pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            print("[Assistant] slow_r18 not available; using torchvision r3d_18.")
            self._backend = "torchvision"
            from torchvision.models.video import r3d_18

            if pretrained:
                try:
                    self.model = r3d_18(weights="DEFAULT")
                except TypeError:
                    self.model = r3d_18(pretrained=True)
            else:
                try:
                    self.model = r3d_18(weights=None)
                except TypeError:
                    self.model = r3d_18(pretrained=False)

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
