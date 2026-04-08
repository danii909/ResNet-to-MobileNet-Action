"""Evaluation metrics: accuracy, model size, inference latency."""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, float]:
    """Compute top-1 and top-5 accuracy on a dataset.

    Returns:
        Dict with 'top1' and 'top5' accuracy percentages.
    """
    model.eval()
    correct_1 = 0
    correct_5 = 0
    total = 0

    for clips, labels in dataloader:
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(clips)

        # Top-1
        preds = logits.argmax(dim=1)
        correct_1 += (preds == labels).sum().item()

        # Top-5
        _, top5 = logits.topk(5, dim=1)
        correct_5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    return {
        "top1": 100.0 * correct_1 / total,
        "top5": 100.0 * correct_5 / total,
    }


def compute_model_size(model: nn.Module) -> dict[str, float]:
    """Compute model parameter count and size.

    Returns:
        Dict with 'param_count', 'param_count_trainable', 'size_mb'.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)

    return {
        "param_count": total,
        "param_count_trainable": trainable,
        "size_mb": size_mb,
    }


@torch.no_grad()
def compute_inference_time(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> dict[str, float]:
    """Measure average inference latency in milliseconds.

    Args:
        model: Model to benchmark.
        input_shape: Shape of a single input tensor (C, T, H, W).
        device: Device to run on.
        num_runs: Number of timed forward passes.
        warmup_runs: Number of warmup passes (not timed).

    Returns:
        Dict with 'avg_ms', 'std_ms', 'min_ms', 'max_ms'.
    """
    model.eval()
    dummy = torch.randn(1, *input_shape, device=device)

    # Warmup
    for _ in range(warmup_runs):
        model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms

    import numpy as np
    times_arr = np.array(times)

    return {
        "avg_ms": float(times_arr.mean()),
        "std_ms": float(times_arr.std()),
        "min_ms": float(times_arr.min()),
        "max_ms": float(times_arr.max()),
    }
