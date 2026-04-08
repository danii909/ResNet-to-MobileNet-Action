"""CLI entry point for training.

Usage:
    python -m src.training.train --config experiments/configs/teacher.yaml
    python -m src.training.train --config experiments/configs/distillation.yaml --override training.lr=0.005
"""

import os
import random
import sys

import numpy as np
import torch

# Force unbuffered output so prints show up in SLURM logs immediately
os.environ.setdefault("PYTHONUNBUFFERED", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from src.datasets.ucf101 import get_dataloaders
from src.models.student import get_student
from src.models.teacher import get_teacher
from src.training.trainer import Trainer
from src.utils import logger
from src.utils.config import get_config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    config = get_config(description="KD Action Recognition Training")
    print("[main] Config loaded.")

    # Seed
    seed = config.get("seed", 42)
    set_seed(seed)
    print(f"[main] Seed set to {seed}.")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[main] Using device: {device}")
    if device.type == "cuda":
        print(f"[main] GPU: {torch.cuda.get_device_name(0)}")

    # Init logging
    print("[main] Initializing W&B logger...")
    logger.init(config)
    print("[main] W&B logger initialized.")

    # Data
    print("[main] Loading dataset...")
    import time
    t0 = time.time()
    dataloaders = get_dataloaders(config)
    print(f"[main] Dataset loaded in {time.time()-t0:.1f}s.")
    print(f"[main] Train batches: {len(dataloaders['train'])}, Test batches: {len(dataloaders['test'])}")
    num_classes = config["dataset"].get("num_classes", 101)

    # Mode
    mode = config["training"]["mode"]
    print(f"[main] Training mode: {mode}")
    print(f"[main] Building model(s)...")

    # Build model(s)
    teacher = None
    if mode == "teacher_finetune":
        model = get_teacher(
            num_classes=num_classes,
            pretrained=config["model"].get("pretrained", True),
            freeze_backbone=config["model"].get("freeze_backbone", False),
        )
    elif mode == "baseline":
        model = get_student(
            num_classes=num_classes,
            width_mult=config["model"].get("width_mult", 1.0),
        )
    elif mode in ("distillation", "distillation_at"):
        # Student
        extract_feats = (mode == "distillation_at")
        model = get_student(
            num_classes=num_classes,
            width_mult=config["model"].get("width_mult", 1.0),
            extract_features=extract_feats,
        )
        # Teacher (frozen, eval mode)
        teacher_ckpt = config["distillation"]["teacher_checkpoint"]
        teacher = get_teacher(
            num_classes=num_classes,
            pretrained=False,
            extract_features=extract_feats,
            checkpoint_path=teacher_ckpt,
        )
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    # Log model info
    param_count = sum(p.numel() for p in model.parameters())
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    logger.log_model_info("model", param_count, size_mb)
    print(f"[main] Model: {param_count:,} parameters ({size_mb:.1f} MB)")

    if teacher is not None:
        t_params = sum(p.numel() for p in teacher.parameters())
        t_size = sum(p.numel() * p.element_size() for p in teacher.parameters()) / (1024 ** 2)
        logger.log_model_info("teacher", t_params, t_size)
        print(f"[main] Teacher: {t_params:,} parameters ({t_size:.1f} MB)")
        print(f"[main] Compression ratio: {t_params / param_count:.1f}x")

    # Train
    print("[main] Starting trainer...")
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=dataloaders["train"],
        test_loader=dataloaders["test"],
        teacher=teacher,
        device=device,
    )
    results = trainer.train()

    # Finish logging
    logger.finish()
    print(f"Done. Best accuracy: {results['best_acc']:.2f}%")
    if "best_epoch" in results:
        print(f"Best epoch: {results['best_epoch'] + 1}")
    if "run_log_dir" in results:
        print(f"Structured logs: {results['run_log_dir']}")


if __name__ == "__main__":
    main()
