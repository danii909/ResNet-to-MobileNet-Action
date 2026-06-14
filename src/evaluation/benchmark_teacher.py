"""Teacher (3D ResNet-50) inference latency benchmark.

Measures forward-pass latency and throughput of the fine-tuned teacher
on the UCF-101 test set across multiple batch sizes (GPU + CPU), and
prints a side-by-side comparison with the best student (24f, KD T=20).

Usage:
    python -m src.evaluation.benchmark_teacher

Output format mirrors benchmark_comparison.py so results are directly
comparable to the existing slurm-benchmark-4683.log data.
"""

import time
import torch
import torch.nn as nn
from pathlib import Path

from src.models.teacher import get_teacher
from src.models.student import get_student


@torch.no_grad()
def benchmark_model(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    batch_size: int = 1,
    num_runs: int = 100,
    warmup_runs: int = 20,
) -> dict:
    """Measure latency and throughput with random synthetic input.
    
    Uses the same protocol as benchmark_comparison.py:
    - CUDA synchronize before/after each timed call.
    - time.perf_counter() for high-resolution wall-clock timing.
    """
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
        times.append((end - start) * 1000.0)  # ms per batch

    times_t = torch.tensor(times)
    avg_ms = float(times_t.mean())
    std_ms = float(times_t.std())
    min_ms = float(times_t.min())
    max_ms = float(times_t.max())
    throughput = (batch_size * 1000.0) / avg_ms  # clips / sec

    return {
        "avg": avg_ms,
        "std": std_ms,
        "min": min_ms,
        "max": max_ms,
        "throughput": throughput,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  TEACHER vs STUDENT — LATENCY BENCHMARK")
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU:    {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    print()

    # ------------------------------------------------------------------ #
    # Checkpoint paths (cluster layout)
    # ------------------------------------------------------------------ #
    CHECKPOINT_ROOT = Path("/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints")

    teacher_ckpt = CHECKPOINT_ROOT / "teacher_finetune_best.pth"
    student_ckpt = CHECKPOINT_ROOT / "slurm-train-eval-4565/kd_t20_a07_24f_lightaug/distillation_best.pth"

    # Local fallbacks (dev machine)
    if not teacher_ckpt.exists():
        teacher_ckpt = Path("experiments/checkpoints/teacher_finetune_best.pth")
    if not student_ckpt.exists():
        student_ckpt = Path("experiments/checkpoints/distillation_best.pth")

    # ------------------------------------------------------------------ #
    # Load models
    # ------------------------------------------------------------------ #
    print(f"Loading Teacher (3D ResNet-50)...")
    print(f"  Checkpoint: {teacher_ckpt}")
    try:
        teacher = get_teacher(
            num_classes=101,
            pretrained=False,
            checkpoint_path=str(teacher_ckpt),
        ).to(device)
        print("  Teacher loaded successfully.")
    except Exception as e:
        print(f"  WARNING: {e}. Falling back to Kinetics-pretrained weights only.")
        teacher = get_teacher(num_classes=101, pretrained=True).to(device)

    print()
    print(f"Loading Student (MobileNet3D, 24f, KD T=20)...")
    print(f"  Checkpoint: {student_ckpt}")
    try:
        student = get_student(
            num_classes=101,
            width_mult=1.0,
            checkpoint_path=str(student_ckpt),
        ).to(device)
        print("  Student loaded successfully.")
    except Exception as e:
        print(f"  WARNING: {e}. Using random student weights.")
        student = get_student(num_classes=101, width_mult=1.0).to(device)

    # Input shapes
    #   Teacher (slow_r50): expects [B, C, T, H, W] with T=8 at stride-8
    #   (UCF fine-tuning used 8 frames for slow_r50, matching Kinetics convention)
    #   Student: 24 frames, 112x112 crop
    shape_teacher = (3, 8, 112, 112)
    shape_student = (3, 24, 112, 112)

    batch_sizes = [1, 4, 8, 16, 32, 64]

    # ------------------------------------------------------------------ #
    # GPU Benchmark
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("  GPU BENCHMARK")
    print("=" * 60)
    print(
        f"{'BS':<5} | {'Teacher (ResNet-50)':<24} | {'Student (MobileNet3D)':<24} | {'Speedup':<10} | {'Student faster %':<16}"
    )
    print("-" * 88)

    for bs in batch_sizes:
        res_t = benchmark_model(teacher, shape_teacher, device, batch_size=bs)
        res_s = benchmark_model(student, shape_student, device, batch_size=bs)

        speedup = res_t["avg"] / res_s["avg"]
        saved_pct = (res_t["avg"] - res_s["avg"]) / res_t["avg"] * 100.0

        print(
            f"{bs:<5} | "
            f"{res_t['avg']:>7.2f} ms ({res_t['throughput']:>6.1f}/s) | "
            f"{res_s['avg']:>7.2f} ms ({res_s['throughput']:>6.1f}/s) | "
            f"{speedup:>8.2f}x | "
            f"{saved_pct:>10.2f}%"
        )

    # ------------------------------------------------------------------ #
    # CPU Benchmark (BS 1 and 8 only, limited runs to keep it fast)
    # ------------------------------------------------------------------ #
    print()
    print("=" * 60)
    print("  CPU BENCHMARK (BS: 1 & 8, 20 runs each)")
    print("=" * 60)

    teacher_cpu = teacher.to("cpu")
    student_cpu = student.to("cpu")

    print(
        f"{'BS':<9} | {'Teacher (ResNet-50)':<24} | {'Student (MobileNet3D)':<24} | {'Speedup':<10} | {'Student faster %':<16}"
    )
    print("-" * 92)

    for bs in [1, 8]:
        res_t = benchmark_model(
            teacher_cpu, shape_teacher, torch.device("cpu"),
            batch_size=bs, num_runs=20, warmup_runs=5
        )
        res_s = benchmark_model(
            student_cpu, shape_student, torch.device("cpu"),
            batch_size=bs, num_runs=20, warmup_runs=5
        )

        speedup = res_t["avg"] / res_s["avg"]
        saved_pct = (res_t["avg"] - res_s["avg"]) / res_t["avg"] * 100.0

        print(
            f"CPU BS {bs:<3} | "
            f"{res_t['avg']:>7.2f} ms ({res_t['throughput']:>6.1f}/s) | "
            f"{res_s['avg']:>7.2f} ms ({res_s['throughput']:>6.1f}/s) | "
            f"{speedup:>8.2f}x | "
            f"{saved_pct:>10.2f}%"
        )

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
