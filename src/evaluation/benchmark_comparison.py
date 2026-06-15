"""Inference benchmark comparison script.

Compares the inference latency and throughput of:
1. 24-frame student model (slurm-train-eval-4565)
2. 16-frame student model (slurm-train-eval-4675)

Usage:
    python -m src.evaluation.benchmark_comparison
"""

import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

from src.models.student import get_student


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    input_shape: tuple[int, ...],
    device: torch.device,
    batch_size: int = 1,
    num_runs: int = 100,
    warmup_runs: int = 20,
) -> dict[str, float]:
    """Measure latency and throughput."""
    model.eval()
    dummy = torch.randn(batch_size, *input_shape, device=device)

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

    times_arr = torch.tensor(times)
    avg_ms = float(times_arr.mean())
    std_ms = float(times_arr.std())
    min_ms = float(times_arr.min())
    max_ms = float(times_arr.max())
    
    # Throughput: clips/sec
    throughput = (batch_size * 1000.0) / avg_ms

    return {
        "avg": avg_ms,
        "std": std_ms,
        "min": min_ms,
        "max": max_ms,
        "throughput": throughput,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"BENCHMARK LATENCY COMPARISON")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"============================================================\n")

    # Paths on cluster
    ckpt_a = Path("/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/slurm-train-eval-4565/kd_t20_a07_24f_lightaug/distillation_best.pth")
    ckpt_b = Path("/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/slurm-train-eval-4675/distillation_best.pth")

    # Fallback to local paths if running locally
    if not ckpt_a.exists():
        ckpt_a = Path("experiments/checkpoints/distillation_best.pth")  # local mock/fallback if needed
    if not ckpt_b.exists():
        ckpt_b = Path("experiments/checkpoints/distillation_best.pth")

    print(f"Loading Model A (24f)...")
    print(f"Path: {ckpt_a}")
    try:
        model_a = get_student(num_classes=101, width_mult=1.0, checkpoint_path=str(ckpt_a))
        model_a = model_a.to(device)
        print("Model A loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Model A weights ({e}). Initializing random weights.")
        model_a = get_student(num_classes=101, width_mult=1.0).to(device)

    print(f"\nLoading Model B (16f)...")
    print(f"Path: {ckpt_b}")
    try:
        model_b = get_student(num_classes=101, width_mult=1.0, checkpoint_path=str(ckpt_b))
        model_b = model_b.to(device)
        print("Model B loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load Model B weights ({e}). Initializing random weights.")
        model_b = get_student(num_classes=101, width_mult=1.0).to(device)

    batch_sizes = [1, 4, 8, 16, 32, 64]
    
    # Target configurations
    # Model A: 24 frames, crop 112
    shape_a = (3, 24, 112, 112)
    # Model B: 16 frames, crop 112
    shape_b = (3, 16, 112, 112)

    print(f"\nBenchmarking over batch sizes: {batch_sizes}...")
    print(f"Model A Input Shape: {shape_a}")
    print(f"Model B Input Shape: {shape_b}\n")

    results_table = []
    
    # Column headers
    print(f"{'BS':<5} | {'Model A (24f)':<22} | {'Model B (16f)':<22} | {'Speedup':<10} | {'Saved %':<10}")
    print("-" * 78)

    for bs in batch_sizes:
        # Benchmark Model A
        res_a = benchmark_model(model_a, shape_a, device, batch_size=bs)
        # Benchmark Model B
        res_b = benchmark_model(model_b, shape_b, device, batch_size=bs)

        # Speedup/Saved
        speedup = res_a["avg"] / res_b["avg"]
        saved_pct = (res_a["avg"] - res_b["avg"]) / res_a["avg"] * 100.0

        print(
            f"{bs:<5} | "
            f"{res_a['avg']:>6.2f} ms ({res_a['throughput']:>6.1f}/s) | "
            f"{res_b['avg']:>6.2f} ms ({res_b['throughput']:>6.1f}/s) | "
            f"{speedup:>7.2f}x | "
            f"{saved_pct:>7.2f}%"
        )
        
        results_table.append({
            "bs": bs,
            "avg_a": res_a["avg"],
            "thru_a": res_a["throughput"],
            "avg_b": res_b["avg"],
            "thru_b": res_b["throughput"],
            "speedup": speedup,
            "saved": saved_pct
        })

    # Run CPU benchmark for batch sizes 1 and 8 when the primary device is CUDA, to report CPU performance separately.
    if device.type == "cuda":
        print(f"\n============================================================")
        print(f"CPU BENCHMARK (Batch Size: 1 & 8)")
        print(f"============================================================")
        model_a_cpu = model_a.to("cpu")
        model_b_cpu = model_b.to("cpu")
        
        for bs in [1, 8]:
            res_a = benchmark_model(model_a_cpu, shape_a, torch.device("cpu"), batch_size=bs, num_runs=20, warmup_runs=5)
            res_b = benchmark_model(model_b_cpu, shape_b, torch.device("cpu"), batch_size=bs, num_runs=20, warmup_runs=5)
            speedup = res_a["avg"] / res_b["avg"]
            saved_pct = (res_a["avg"] - res_b["avg"]) / res_a["avg"] * 100.0
            print(
                f"CPU BS {bs:<2} | "
                f"{res_a['avg']:>7.2f} ms ({res_a['throughput']:>6.1f}/s) | "
                f"{res_b['avg']:>7.2f} ms ({res_b['throughput']:>6.1f}/s) | "
                f"{speedup:>7.2f}x | "
                f"{saved_pct:>7.2f}%"
            )


if __name__ == "__main__":
    main()
