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
from src.models.assistant import get_assistant
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
    eval_loader = dataloaders.get("eval", dataloaders["test"])
    print(
        f"[main] Train batches: {len(dataloaders['train'])}, "
        f"Eval batches: {len(eval_loader)}, Test batches: {len(dataloaders['test'])}"
    )
    num_classes = config["dataset"].get("num_classes", 101)

    # Mode
    mode = config["training"]["mode"]
    print(f"[main] Training mode: {mode}")
    print(f"[main] Building model(s)...")

    model_cfg = config.get("model", {})
    model_type = model_cfg.get("type", "student")

    # Build model(s)
    teacher = None
    if mode == "teacher_finetune":
        if model_type == "teacher":
            model = get_teacher(
                num_classes=num_classes,
                pretrained=model_cfg.get("pretrained", True),
                freeze_backbone=model_cfg.get("freeze_backbone", False),
            )
        elif model_type == "assistant":
            model = get_assistant(
                num_classes=num_classes,
                pretrained=model_cfg.get("pretrained", False),
                freeze_backbone=model_cfg.get("freeze_backbone", False),
            )
        else:
            raise ValueError(f"Unknown model type for teacher_finetune: {model_type}")
    elif mode == "baseline":
        if model_type == "student":
            model = get_student(
                num_classes=num_classes,
                width_mult=model_cfg.get("width_mult", 1.0),
            )
        elif model_type == "assistant":
            model = get_assistant(
                num_classes=num_classes,
                pretrained=model_cfg.get("pretrained", False),
                freeze_backbone=model_cfg.get("freeze_backbone", False),
            )
        else:
            raise ValueError(f"Unknown model type for baseline: {model_type}")
    elif mode in ("distillation", "distillation_at"):
        extract_feats = (mode == "distillation_at")
        if model_type == "student":
            model = get_student(
                num_classes=num_classes,
                width_mult=model_cfg.get("width_mult", 1.0),
                extract_features=extract_feats,
            )
        elif model_type == "assistant":
            model = get_assistant(
                num_classes=num_classes,
                pretrained=model_cfg.get("pretrained", False),
                freeze_backbone=model_cfg.get("freeze_backbone", False),
                extract_features=extract_feats,
            )
        else:
            raise ValueError(f"Unknown model type for distillation: {model_type}")

        dist_cfg = config.get("distillation", {})
        teacher_ckpt = dist_cfg["teacher_checkpoint"]
        teacher_type = dist_cfg.get("teacher_type", "teacher")
        if teacher_type == "teacher":
            teacher = get_teacher(
                num_classes=num_classes,
                pretrained=False,
                extract_features=extract_feats,
                checkpoint_path=teacher_ckpt,
            )
        elif teacher_type == "assistant":
            teacher = get_assistant(
                num_classes=num_classes,
                pretrained=False,
                extract_features=extract_feats,
                checkpoint_path=teacher_ckpt,
            )
        elif teacher_type == "student":
            # Allow student architecture to act as teacher (student->student distillation)
            teacher = get_student(
                num_classes=num_classes,
                width_mult=dist_cfg.get("teacher_width_mult", 1.0),
                extract_features=extract_feats,
                checkpoint_path=teacher_ckpt,
            )
        else:
            raise ValueError(f"Unknown distillation teacher type: {teacher_type}")
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
        eval_loader=eval_loader,
        teacher=teacher,
        device=device,
    )
    results = trainer.train()

    # Finish logging
    logger.finish()
    print(f"Done. Best eval accuracy: {results['best_acc']:.2f}%")
    if "best_epoch" in results:
        print(f"Best epoch: {results['best_epoch'] + 1}")
    if "run_log_dir" in results:
        print(f"Structured logs: {results['run_log_dir']}")


if __name__ == "__main__":
    main()
