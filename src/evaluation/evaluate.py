"""CLI entry point for evaluation.

Usage:
    python -m src.evaluation.evaluate --config experiments/configs/baseline.yaml --checkpoint experiments/checkpoints/baseline_best.pth
    python -m src.evaluation.evaluate --config experiments/configs/teacher.yaml --checkpoint experiments/checkpoints/teacher_finetune_best.pth
"""

import json
import os
from datetime import datetime
from pathlib import Path

import torch

from src.datasets.ucf101 import get_dataloaders
from src.evaluation.metrics import compute_accuracy, compute_inference_time, compute_model_size
from src.models.student import get_student
from src.models.teacher import get_teacher
from src.utils import logger
from src.utils.config import get_config


def _build_eval_log_dir() -> Path:
    """Create a deterministic directory for evaluation artifacts."""
    explicit_dir = os.environ.get("EVAL_LOG_DIR")
    if explicit_dir:
        run_dir = Path(explicit_dir)
    else:
        root_dir = Path(os.environ.get("EVAL_LOG_ROOT", "experiments/logs"))
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id:
            run_name = f"slurm-eval-{job_id}"
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            run_name = f"local-eval-{timestamp}"
        run_dir = root_dir / run_name

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_eval_summary(
    eval_log_dir: Path,
    *,
    config: dict,
    model_type: str,
    checkpoint: str,
    device: torch.device,
    acc: dict[str, float],
    size_info: dict[str, float],
    timing: dict[str, float],
) -> None:
    """Persist a structured evaluation summary next to the SLURM logs."""
    finished_at = datetime.now()
    summary = {
        "mode": config.get("training", {}).get("mode", "evaluation"),
        "model_type": model_type,
        "checkpoint": checkpoint,
        "device": str(device),
        "top1": acc["top1"],
        "top5": acc["top5"],
        "param_count": size_info["param_count"],
        "param_count_trainable": size_info["param_count_trainable"],
        "size_mb": size_info["size_mb"],
        "inference_avg_ms": timing["avg_ms"],
        "inference_std_ms": timing["std_ms"],
        "inference_min_ms": timing["min_ms"],
        "inference_max_ms": timing["max_ms"],
        "finished_at": finished_at.isoformat(timespec="seconds"),
    }

    txt_path = eval_log_dir / "evaluation_summary.txt"
    json_path = eval_log_dir / "evaluation_summary.json"

    lines = [
        "Evaluation Summary",
        "=" * 40,
        f"mode: {summary['mode']}",
        f"model_type: {summary['model_type']}",
        f"checkpoint: {summary['checkpoint']}",
        f"device: {summary['device']}",
        f"top1: {summary['top1']:.2f}",
        f"top5: {summary['top5']:.2f}",
        f"param_count: {summary['param_count']}",
        f"param_count_trainable: {summary['param_count_trainable']}",
        f"size_mb: {summary['size_mb']:.2f}",
        f"inference_avg_ms: {summary['inference_avg_ms']:.2f}",
        f"inference_std_ms: {summary['inference_std_ms']:.2f}",
        f"inference_min_ms: {summary['inference_min_ms']:.2f}",
        f"inference_max_ms: {summary['inference_max_ms']:.2f}",
        f"finished_at: {summary['finished_at']}",
    ]

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Eval] Structured summary written to: {txt_path}")
    print(f"[Eval] Structured summary written to: {json_path}")


def main() -> None:
    config = get_config(description="KD Action Recognition Evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_log_dir = _build_eval_log_dir()
    print(f"Evaluation log dir: {eval_log_dir}")

    # Init logging
    logger.init(config)

    # Data
    dataloaders = get_dataloaders(config)
    num_classes = config["dataset"].get("num_classes", 101)
    ds_cfg = config["dataset"]

    # Determine model type from config
    model_type = config["model"]["type"]
    checkpoint = config.get("evaluation", {}).get("checkpoint")
    if checkpoint is None:
        raise ValueError("evaluation.checkpoint must be set in the config or via --override")

    if model_type == "teacher":
        model = get_teacher(
            num_classes=num_classes,
            pretrained=False,
            checkpoint_path=checkpoint,
        )
    elif model_type == "student":
        model = get_student(
            num_classes=num_classes,
            width_mult=config["model"].get("width_mult", 1.0),
            checkpoint_path=checkpoint,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    print(f"Model type: {model_type}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Inference target: {model.__class__.__name__}")

    # --- Accuracy ---
    print("\nComputing accuracy...")
    use_amp = config.get("training", {}).get("mixed_precision", True)
    acc = compute_accuracy(model, dataloaders["test"], device, use_amp=use_amp)
    print(f"  Top-1 Accuracy: {acc['top1']:.2f}%")
    print(f"  Top-5 Accuracy: {acc['top5']:.2f}%")

    # --- Model Size ---
    size_info = compute_model_size(model)
    print("\nModel Size:")
    print(f"  Parameters: {size_info['param_count']:,}")
    print(f"  Trainable:  {size_info['param_count_trainable']:,}")
    print(f"  Size:       {size_info['size_mb']:.2f} MB")

    # --- Inference Time ---
    num_frames = ds_cfg.get("num_frames", 16)
    crop_size = ds_cfg.get("crop_size", 112)
    input_shape = (3, num_frames, crop_size, crop_size)

    print(f"\nBenchmarking inference (input shape: {input_shape})...")
    timing = compute_inference_time(model, input_shape, device)
    print(f"  Avg: {timing['avg_ms']:.2f} ms")
    print(f"  Std: {timing['std_ms']:.2f} ms")
    print(f"  Min: {timing['min_ms']:.2f} ms")
    print(f"  Max: {timing['max_ms']:.2f} ms")

    # --- Log to W&B ---
    results = {
        "eval/top1": acc["top1"],
        "eval/top5": acc["top5"],
        "eval/param_count": size_info["param_count"],
        "eval/size_mb": size_info["size_mb"],
        "eval/inference_avg_ms": timing["avg_ms"],
    }
    logger.log_metrics(results)

    _write_eval_summary(
        eval_log_dir,
        config=config,
        model_type=model_type,
        checkpoint=checkpoint,
        device=device,
        acc=acc,
        size_info=size_info,
        timing=timing,
    )

    # --- Summary Table ---
    print("\n" + "=" * 60)
    print(f"{'Metric':<30} {'Value':>20}")
    print("-" * 60)
    print(f"{'Top-1 Accuracy (%)':<30} {acc['top1']:>20.2f}")
    print(f"{'Top-5 Accuracy (%)':<30} {acc['top5']:>20.2f}")
    print(f"{'Parameters':<30} {size_info['param_count']:>20,}")
    print(f"{'Model Size (MB)':<30} {size_info['size_mb']:>20.2f}")
    print(f"{'Inference Time (ms)':<30} {timing['avg_ms']:>20.2f}")
    print("=" * 60)

    logger.finish()


if __name__ == "__main__":
    main()
