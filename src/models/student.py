"""Student model: MobileNet3D (MobileNetV2 architecture adapted to 3D convolutions).

Lightweight model for action recognition with ~3-5M parameters,
targeting a 5-10x reduction compared to the 3D ResNet-50 teacher.
"""

from typing import Optional

import torch
import torch.nn as nn


def _make_divisible(v: float, divisor: int, min_value: int = 8) -> int:
    """Ensure channel count is divisible by divisor."""
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBnReLU3d(nn.Sequential):
    """Conv3d + BatchNorm3d + ReLU6 block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int, int] = (1, 1, 1),
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: tuple[int, int, int] = (0, 0, 0),
        groups: int = 1,
        activation: bool = True,
    ):
        layers = [
            nn.Conv3d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, groups=groups, bias=False,
            ),
            nn.BatchNorm3d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU6(inplace=True))
        super().__init__(*layers)


class InvertedResidual3D(nn.Module):
    """MobileNetV2 inverted residual block adapted to 3D.

    Architecture: pointwise expand → depthwise 3D conv → pointwise project.
    Skip connection when stride=1 and in_channels == out_channels.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Spatial stride (temporal stride is always 1).
        expand_ratio: Expansion factor for the hidden dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 1,
    ):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expand_ratio))

        layers: list[nn.Module] = []

        # Pointwise expansion (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers.append(ConvBnReLU3d(in_channels, hidden_dim, kernel_size=(1, 1, 1)))

        # Depthwise 3D convolution
        layers.append(
            ConvBnReLU3d(
                hidden_dim, hidden_dim,
                kernel_size=(3, 3, 3),
                stride=(1, stride, stride),
                padding=(1, 1, 1),
                groups=hidden_dim,
            )
        )

        # Pointwise linear projection (no activation)
        layers.append(
            ConvBnReLU3d(hidden_dim, out_channels, kernel_size=(1, 1, 1), activation=False)
        )

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class MobileNet3D(nn.Module):
    """MobileNetV2 architecture adapted to 3D video inputs.

    Input shape: [B, 3, T, H, W] where T=num_frames, H=W=crop_size.
    Output: [B, num_classes] logits.

    Args:
        num_classes: Number of output classes.
        width_mult: Width multiplier for channel count scaling.
        extract_features: If True, store intermediate activations for AT.
    """

    # MobileNetV2 inverted residual settings:
    # (expand_ratio, output_channels, num_blocks, stride)
    INVERTED_RESIDUAL_SETTINGS = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ]

    # Stage indices to capture for attention transfer (matching teacher blocks 3,4,5)
    FEATURE_STAGES = [2, 4, 6]

    def __init__(
        self,
        num_classes: int = 101,
        width_mult: float = 1.0,
        extract_features: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.extract_features = extract_features
        self._intermediate_features: dict[int, torch.Tensor] = {}

        input_channels = _make_divisible(32 * width_mult, 8)
        last_channels = _make_divisible(1280 * width_mult, 8)

        # Initial convolution
        self.stem = ConvBnReLU3d(
            3, input_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )

        # Inverted residual stages
        self.stages = nn.ModuleList()
        for t, c, n, s in self.INVERTED_RESIDUAL_SETTINGS:
            output_channels = _make_divisible(c * width_mult, 8)
            blocks = []
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(
                    InvertedResidual3D(input_channels, output_channels, stride, t)
                )
                input_channels = output_channels
            self.stages.append(nn.Sequential(*blocks))

        # Final expansion layer
        self.final_conv = ConvBnReLU3d(input_channels, last_channels, kernel_size=(1, 1, 1))

        # Classifier
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(last_channels, num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, T, H, W].

        Returns:
            Logits [B, num_classes].
        """
        self._intermediate_features.clear()
        x = self.stem(x)

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.extract_features and stage_idx in self.FEATURE_STAGES:
                self._intermediate_features[stage_idx] = x

        x = self.final_conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits

    def get_intermediate_features(self) -> dict[int, torch.Tensor]:
        """Return intermediate features captured during the last forward pass."""
        return self._intermediate_features

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pre-classifier embedding.

        Returns:
            Embedding tensor of shape [B, last_channels].
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.final_conv(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


def get_student(
    num_classes: int = 101,
    width_mult: float = 1.0,
    extract_features: bool = False,
    checkpoint_path: Optional[str] = None,
) -> MobileNet3D:
    """Factory function to create and optionally load a student model."""
    model = MobileNet3D(
        num_classes=num_classes,
        width_mult=width_mult,
        extract_features=extract_features,
    )

    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)

    return model
