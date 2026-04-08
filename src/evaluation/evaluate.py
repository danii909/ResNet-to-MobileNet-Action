"""CLI entry point for evaluation.

Usage:
    python -m src.evaluation.evaluate --config experiments/configs/baseline.yaml --checkpoint experiments/checkpoints/baseline_best.pth
    python -m src.evaluation.evaluate --config experiments/configs/teacher.yaml --checkpoint experiments/checkpoints/teacher_finetune_best.pth
"""

import torch

from src.datasets.ucf101 import get_dataloaders
from src.evaluation.metrics import compute_accuracy, compute_inference_time, compute_model_size
from src.models.student import get_student
from src.models.teacher import get_teacher
from src.utils import logger
from src.utils.config import get_config


def main() -> None:
    config = get_config(description="KD Action Recognition Evaluation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
