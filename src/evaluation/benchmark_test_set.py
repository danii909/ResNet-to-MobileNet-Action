"""Test Set inference benchmark comparison script.

Compares the inference latency of:
1. 24-frame student model
2. 16-frame student model

On the actual UCF-101 Test Set, measuring ONLY the model(clips) time
to avoid dataloader/decoding overhead.

Usage:
    python -m src.evaluation.benchmark_test_set
"""

import time
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from src.models.student import get_student
from src.datasets.ucf101 import get_dataloaders


@torch.no_grad()
def benchmark_model_on_test_set(
    model: nn.Module,
    dataloader,
    device: torch.device,
    desc: str = "Benchmarking",
    warmup_batches: int = 2,
    max_batches: int = None,
) -> dict[str, float]:
    """Measure latency and throughput on the actual test set.
    Times ONLY the `model(clips)` call.
    """
    model.eval()
    
    # Warmup
    # We take a few batches from the dataloader for warmup
    warmup_clips = []
    for i, (clips, _) in enumerate(dataloader):
        warmup_clips.append(clips.to(device, non_blocking=True))
        if i >= warmup_batches - 1:
            break
            
    for clips in warmup_clips:
        if device.type == "cuda":
            torch.cuda.synchronize()
        model(clips)
        if device.type == "cuda":
            torch.cuda.synchronize()

    times = []
    total_clips = 0
    
    pbar = tqdm(dataloader, desc=desc, leave=True)
    for i, (clips, labels) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        clips = clips.to(device, non_blocking=True)
        batch_size = clips.size(0)
        total_clips += batch_size
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        # --- TIMED SECTION ---
        model(clips)
        # ---------------------
        
        if device.type == "cuda":
            torch.cuda.synchronize()
            
        end = time.perf_counter()
        
        # We record the time taken for the WHOLE batch
        # To get per-clip latency later or keep per-batch latency.
        # We will store per-batch time in ms.
        batch_time_ms = (end - start) * 1000.0
        times.append(batch_time_ms / batch_size)  # ms per clip in this batch

    times_tensor = torch.tensor(times)
    
    avg_ms_per_clip = float(times_tensor.mean())
    std_ms_per_clip = float(times_tensor.std())
    
    # Throughput (clips/sec) based on total time vs total clips
    # average time per clip in ms -> 1000 / avg_ms = clips per sec
    throughput = 1000.0 / avg_ms_per_clip

    return {
        "avg_ms_per_clip": avg_ms_per_clip,
        "std_ms_per_clip": std_ms_per_clip,
        "throughput": throughput,
        "total_clips": total_clips
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"============================================================")
    print(f"REAL TEST SET LATENCY BENCHMARK")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"============================================================\n")

    # Paths on cluster
    ckpt_a = Path("/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/slurm-train-eval-4565/kd_t20_a07_24f_lightaug/distillation_best.pth")
    ckpt_b = Path("/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/slurm-train-eval-4675/distillation_best.pth")

    if not ckpt_a.exists():
        ckpt_a = Path("experiments/checkpoints/distillation_best.pth")
    if not ckpt_b.exists():
        ckpt_b = Path("experiments/checkpoints/distillation_best.pth")

    print(f"Loading Model A (24f)...")
    try:
        model_a = get_student(num_classes=101, width_mult=1.0, checkpoint_path=str(ckpt_a)).to(device)
    except Exception as e:
        print(f"Warning: {e}. Using random weights.")
        model_a = get_student(num_classes=101, width_mult=1.0).to(device)

    print(f"Loading Model B (16f)...")
    try:
        model_b = get_student(num_classes=101, width_mult=1.0, checkpoint_path=str(ckpt_b)).to(device)
    except Exception as e:
        print(f"Warning: {e}. Using random weights.")
        model_b = get_student(num_classes=101, width_mult=1.0).to(device)

    batch_sizes = [1, 8, 16, 32]
    
    print("\nStarting benchmark...")
    
    print(f"{'BS':<5} | {'Model A (24f) Avg ms/clip':<25} | {'Model B (16f) Avg ms/clip':<25} | {'Speedup':<10} | {'Saved %':<10}")
    print("-" * 85)

    for bs in batch_sizes:
        print(f"\nEvaluating Batch Size: {bs}")
        
        # DataLoaders configuration
        cfg_a = {
            "dataset": {
                "backend": "hf",
                "num_frames": 24,
                "crop_size": 112,
                "use_eval_split": False, # Just get test set
            },
            "training": {
                "batch_size": bs,
                "num_workers": 4
            }
        }
        
        cfg_b = {
            "dataset": {
                "backend": "hf",
                "num_frames": 16,
                "crop_size": 112,
                "use_eval_split": False,
            },
            "training": {
                "batch_size": bs,
                "num_workers": 4
            }
        }

        print("  Initializing dataloaders...")
        loader_a = get_dataloaders(cfg_a)["test"]
        loader_b = get_dataloaders(cfg_b)["test"]

        res_a = benchmark_model_on_test_set(model_a, loader_a, device, desc=f"Model A (24f, BS={bs})", warmup_batches=1)
        res_b = benchmark_model_on_test_set(model_b, loader_b, device, desc=f"Model B (16f, BS={bs})", warmup_batches=1)

        speedup = res_a["avg_ms_per_clip"] / res_b["avg_ms_per_clip"]
        saved_pct = (res_a["avg_ms_per_clip"] - res_b["avg_ms_per_clip"]) / res_a["avg_ms_per_clip"] * 100.0

        print(f"\nResults for BS={bs}:")
        print(
            f"{bs:<5} | "
            f"{res_a['avg_ms_per_clip']:>6.2f} ms ({res_a['throughput']:>6.1f}/s) | "
            f"{res_b['avg_ms_per_clip']:>6.2f} ms ({res_b['throughput']:>6.1f}/s) | "
            f"{speedup:>7.2f}x | "
            f"{saved_pct:>7.2f}%"
        )

    # CPU Latency Comparison for Batch Size 1 and 8
    print(f"\n============================================================")
    print(f"REAL TEST SET CPU BENCHMARK (BS: 1 & 8)")
    print(f"============================================================")
    
    # Move models to CPU
    model_a_cpu = model_a.to("cpu")
    model_b_cpu = model_b.to("cpu")
    
    cpu_batch_sizes = [1, 8]
    cpu_max_batches = {1: 50, 8: 20}  # Limit CPU evaluation batches to keep execution short
    
    print(f"{'BS':<5} | {'Model A (24f) Avg ms/clip':<25} | {'Model B (16f) Avg ms/clip':<25} | {'Speedup':<10} | {'Saved %':<10}")
    print("-" * 85)
    
    for bs in cpu_batch_sizes:
        cfg_a = {
            "dataset": {
                "backend": "hf",
                "num_frames": 24,
                "crop_size": 112,
                "use_eval_split": False,
            },
            "training": {
                "batch_size": bs,
                "num_workers": 4
            }
        }
        cfg_b = {
            "dataset": {
                "backend": "hf",
                "num_frames": 16,
                "crop_size": 112,
                "use_eval_split": False,
            },
            "training": {
                "batch_size": bs,
                "num_workers": 4
            }
        }
        
        loader_a = get_dataloaders(cfg_a)["test"]
        loader_b = get_dataloaders(cfg_b)["test"]
        
        res_a = benchmark_model_on_test_set(
            model_a_cpu, loader_a, torch.device("cpu"), 
            desc=f"Model A CPU (BS={bs})", warmup_batches=1, 
            max_batches=cpu_max_batches[bs]
        )
        res_b = benchmark_model_on_test_set(
            model_b_cpu, loader_b, torch.device("cpu"), 
            desc=f"Model B CPU (BS={bs})", warmup_batches=1, 
            max_batches=cpu_max_batches[bs]
        )
        
        speedup = res_a["avg_ms_per_clip"] / res_b["avg_ms_per_clip"]
        saved_pct = (res_a["avg_ms_per_clip"] - res_b["avg_ms_per_clip"]) / res_a["avg_ms_per_clip"] * 100.0
        
        print(f"\nResults CPU for BS={bs}:")
        print(
            f"CPU {bs:<2} | "
            f"{res_a['avg_ms_per_clip']:>6.2f} ms ({res_a['throughput']:>6.1f}/s) | "
            f"{res_b['avg_ms_per_clip']:>6.2f} ms ({res_b['throughput']:>6.1f}/s) | "
            f"{speedup:>7.2f}x | "
            f"{saved_pct:>7.2f}%"
        )

if __name__ == "__main__":
    main()
