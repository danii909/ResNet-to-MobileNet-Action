"""Temporal sub-sampling utilities for Cross-Frame Knowledge Distillation.

In Cross-Frame KD, the Teacher processes the full temporal context (e.g., 24 frames)
while the Student receives a temporally sub-sampled version (e.g., 16 frames).
This forces the Student to learn to approximate the Teacher's rich temporal
representations from a reduced-cost input, maximizing inference throughput.

The sub-sampling is performed entirely on GPU via index_select, avoiding any
CPU↔GPU data transfer overhead during training.

Reference:
    The temporal sub-sampling strategy uses uniform spacing via torch.linspace
    to ensure the sub-sampled clip covers the full temporal span of the original.
"""

import torch


def temporal_subsample(
    clips: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    """Uniformly sub-sample a video batch along the temporal dimension.

    Given a batch of clips with T_teacher frames, extracts num_frames indices
    uniformly spaced across [0, T_teacher - 1] to cover the full temporal span.

    This operation is:
      - Differentiable-safe (uses index_select, no in-place ops on the input)
      - Batched (applies identically to every sample in the batch)
      - Device-agnostic (indices are placed on the same device as the input)

    Args:
        clips: Input tensor of shape [B, C, T, H, W] where T >= num_frames.
        num_frames: Number of frames to extract (must be <= T).

    Returns:
        Sub-sampled tensor of shape [B, C, num_frames, H, W].

    Raises:
        ValueError: If num_frames > T or the input is not 5D.

    Example:
        >>> teacher_clips = torch.randn(4, 3, 24, 112, 112, device='cuda')
        >>> student_clips = temporal_subsample(teacher_clips, num_frames=16)
        >>> student_clips.shape
        torch.Size([4, 3, 16, 112, 112])
    """
    if clips.dim() != 5:
        raise ValueError(
            f"Expected 5D tensor [B, C, T, H, W], got {clips.dim()}D "
            f"with shape {clips.shape}"
        )

    T_teacher = clips.shape[2]

    if num_frames > T_teacher:
        raise ValueError(
            f"Cannot sub-sample {num_frames} frames from a clip with only "
            f"{T_teacher} frames. Ensure student_frames <= teacher_frames."
        )

    # Fast path: no sub-sampling needed
    if num_frames == T_teacher:
        return clips

    # Compute uniformly spaced indices across the full temporal span.
    # linspace(0, T-1, N) gives N points evenly covering [0, T-1],
    # .long() floors to valid frame indices.
    indices = torch.linspace(
        0, T_teacher - 1, num_frames,
        device=clips.device,
    ).long()

    # index_select on dim=2 (temporal) is memory-efficient: it creates a
    # view into the original storage when possible, avoiding a full copy.
    return clips.index_select(dim=2, index=indices)
