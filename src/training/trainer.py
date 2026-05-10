"""Trainer class supporting teacher fine-tuning, baseline student, and knowledge distillation."""

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import CombinedKDATLoss, KDLoss
from src.utils import logger


class Trainer:
    """Unified trainer for all training modes.

    Modes:
        - 'teacher_finetune': Fine-tune the pretrained teacher on UCF-101.
        - 'baseline': Train the student from scratch with CE loss.
        - 'distillation': Train the student with KD loss from the teacher.
        - 'distillation_at': Train the student with KD + Attention Transfer.

    Args:
        config: Full experiment configuration dict.
        model: The model to train (student in distillation modes).
        train_loader: Training DataLoader.
        eval_loader: Evaluation DataLoader.
        teacher: Optional teacher model (required for distillation modes).
        device: Torch device.
    """

    def __init__(
        self,
        config: dict,
        model: nn.Module,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        teacher: Optional[nn.Module] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.config = config
        self.model = model.to(device)
        self.teacher = teacher
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device

        tr_cfg = config["training"]
        self.mode = tr_cfg["mode"]
        self.epochs = tr_cfg["epochs"]
        self.checkpoint_dir = Path(tr_cfg.get("checkpoint_dir", "experiments/checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision
        self.use_amp = tr_cfg.get("mixed_precision", True)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.label_smoothing = tr_cfg.get("label_smoothing", 0.0)
        self.kd_warmup_epochs = tr_cfg.get(
            "kd_warmup_epochs",
            5 if self.mode in ("distillation", "distillation_at") else 0,
        )
        self.current_epoch = 0
        self.hard_criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Setup teacher for distillation
        if self.mode in ("distillation", "distillation_at") and self.teacher is not None:
            self.teacher = self.teacher.to(device)
            self.teacher.eval()

        # Loss function
        self.criterion = self._build_criterion(config)

        # Optimizer
        self.optimizer = self._build_optimizer(tr_cfg)

        # LR scheduler
        self.scheduler = self._build_scheduler(tr_cfg)

        # Gradient clipping
        self.grad_clip = tr_cfg.get("grad_clip", 0.0)

        # Tracking
        self.best_acc = 0.0
        self.start_epoch = 0
        self.best_epoch = -1

        # Structured experiment logs for cluster/local runs
        self.run_started_at = datetime.now()
        self.run_log_dir = self._build_run_log_dir(tr_cfg)
        self.metrics_csv_path = self.run_log_dir / "metrics_epoch.csv"
        self.metrics_jsonl_path = self.run_log_dir / "metrics_epoch.jsonl"
        self.summary_path = self.run_log_dir / "training_summary.txt"
        self.config_summary_path = self.run_log_dir / "config_summary.txt"
        self.config_summary_json_path = self.run_log_dir / "config_summary.json"
        self._init_metrics_files()
        self._write_config_summary()

        # Resume from checkpoint
        resume_path = tr_cfg.get("resume_checkpoint")
        if resume_path and os.path.exists(resume_path):
            self._load_checkpoint(resume_path)

    def _build_run_log_dir(self, tr_cfg: dict) -> Path:
        """Create a deterministic log directory for this run."""
        explicit_dir = tr_cfg.get("run_log_dir")
        if explicit_dir:
            run_dir = Path(explicit_dir)
        else:
            env_dir = os.environ.get("TRAIN_LOG_DIR")
            if env_dir:
                run_dir = Path(env_dir)
            else:
                root_dir = Path(tr_cfg.get("log_root_dir", "experiments/logs"))
                job_id = os.environ.get("SLURM_JOB_ID")
                if job_id:
                    run_name = f"slurm-train-{job_id}"
                else:
                    timestamp = self.run_started_at.strftime("%Y%m%d-%H%M%S")
                    run_name = f"local-train-{timestamp}"
                run_dir = root_dir / run_name

        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _init_metrics_files(self) -> None:
        """Initialize tabular metrics files for downstream plotting."""
        with open(self.metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss",
                "train_acc",
                "eval_loss",
                "eval_acc",
                "eval_acc_top5",
                "lr",
                "best_eval_acc",
            ])

    def _append_epoch_metrics(self, metrics: dict) -> None:
        """Append epoch metrics to CSV + JSONL files."""
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics["epoch"],
                metrics["train_loss"],
                metrics["train_acc"],
                metrics["eval_loss"],
                metrics["eval_acc"],
                metrics["eval_acc_top5"],
                metrics["lr"],
                metrics["best_eval_acc"],
            ])

        with open(self.metrics_jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics) + "\n")

    def _run_profile(self) -> str:
        """Return a high-level profile for this run."""
        if self.mode == "teacher_finetune":
            return "teacher"
        if self.mode == "baseline":
            return "student"
        if self.mode in ("distillation", "distillation_at"):
            return "student_distillation"
        return "unknown"

    def _write_config_summary(self) -> None:
        """Persist a concise config summary with key hyperparameters."""
        model_cfg = self.config.get("model", {})
        data_cfg = self.config.get("dataset", {})
        tr_cfg = self.config.get("training", {})
        kd_cfg = self.config.get("distillation", {})

        profile = self._run_profile()

        summary = {
            "profile": profile,
            "mode": self.mode,
            "seed": self.config.get("seed"),
            "model": {
                "type": model_cfg.get("type"),
                "width_mult": model_cfg.get("width_mult"),
                "pretrained": model_cfg.get("pretrained"),
                "freeze_backbone": model_cfg.get("freeze_backbone"),
            },
            "dataset": {
                "data_dir": data_cfg.get("data_dir"),
                "num_classes": data_cfg.get("num_classes"),
                "num_frames": data_cfg.get("num_frames"),
                "crop_size": data_cfg.get("crop_size"),
                "backend": data_cfg.get("backend"),
                "use_eval_split": data_cfg.get("use_eval_split"),
                "eval_ratio": data_cfg.get("eval_ratio"),
                "split_seed": data_cfg.get("split_seed"),
            },
            "training": {
                "epochs": tr_cfg.get("epochs"),
                "batch_size": tr_cfg.get("batch_size"),
                "num_workers": tr_cfg.get("num_workers"),
                "optimizer": tr_cfg.get("optimizer"),
                "lr": tr_cfg.get("lr"),
                "momentum": tr_cfg.get("momentum"),
                "weight_decay": tr_cfg.get("weight_decay"),
                "scheduler": tr_cfg.get("scheduler"),
                "grad_clip": tr_cfg.get("grad_clip"),
                "mixed_precision": tr_cfg.get("mixed_precision"),
            },
            "distillation": {
                "teacher_type": kd_cfg.get("teacher_type"),
                "teacher_checkpoint": kd_cfg.get("teacher_checkpoint"),
                "temperature": kd_cfg.get("temperature"),
                "alpha": kd_cfg.get("alpha"),
                "at_beta_spatial": kd_cfg.get("at_beta_spatial", kd_cfg.get("at_beta")),
                "at_beta_temporal": kd_cfg.get("at_beta_temporal", kd_cfg.get("at_beta")),
                "teacher_keys": kd_cfg.get("teacher_keys"),
                "student_keys": kd_cfg.get("student_keys"),
            },
        }

        lines = [
            "Config Summary",
            "=" * 40,
            f"profile: {summary['profile']}",
            f"mode: {summary['mode']}",
            f"seed: {summary['seed']}",
            "",
            "[model]",
            f"type: {summary['model']['type']}",
        ]

        model_type = summary["model"]["type"]
        if model_type in ("teacher", "assistant"):
            lines.extend([
                f"pretrained: {summary['model']['pretrained']}",
                f"freeze_backbone: {summary['model']['freeze_backbone']}",
            ])
        else:
            lines.append(f"width_mult: {summary['model']['width_mult']}")

        lines.extend([
            "",
            "[dataset]",
            f"data_dir: {summary['dataset']['data_dir']}",
            f"num_classes: {summary['dataset']['num_classes']}",
            f"num_frames: {summary['dataset']['num_frames']}",
            f"crop_size: {summary['dataset']['crop_size']}",
            f"backend: {summary['dataset']['backend']}",
            f"use_eval_split: {summary['dataset']['use_eval_split']}",
            f"eval_ratio: {summary['dataset']['eval_ratio']}",
            f"split_seed: {summary['dataset']['split_seed']}",
            "",
            "[training]",
            f"epochs: {summary['training']['epochs']}",
            f"batch_size: {summary['training']['batch_size']}",
            f"num_workers: {summary['training']['num_workers']}",
            f"optimizer: {summary['training']['optimizer']}",
            f"lr: {summary['training']['lr']}",
            f"momentum: {summary['training']['momentum']}",
            f"weight_decay: {summary['training']['weight_decay']}",
            f"scheduler: {summary['training']['scheduler']}",
            f"grad_clip: {summary['training']['grad_clip']}",
            f"mixed_precision: {summary['training']['mixed_precision']}",
        ])

        if profile == "student_distillation":
            lines.extend([
                "",
                "[distillation]",
                f"teacher_type: {summary['distillation']['teacher_type']}",
                f"teacher_checkpoint: {summary['distillation']['teacher_checkpoint']}",
                f"temperature: {summary['distillation']['temperature']}",
                f"alpha: {summary['distillation']['alpha']}",
            ])
            if self.mode == "distillation_at":
                lines.extend([
                    f"at_beta_spatial: {summary['distillation']['at_beta_spatial']}",
                    f"at_beta_temporal: {summary['distillation']['at_beta_temporal']}",
                    f"teacher_keys: {summary['distillation']['teacher_keys']}",
                    f"student_keys: {summary['distillation']['student_keys']}",
                ])

        with open(self.config_summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        with open(self.config_summary_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    def _write_ls_markers(self, final_metrics: dict) -> None:
        """Write empty marker files whose names expose key KPIs in `ls -la`."""
        for old_file in self.run_log_dir.glob("metric_*"):
            old_file.unlink(missing_ok=True)

        marker_names = [
            f"metric_best_eval_acc_{self.best_acc:.2f}",
            f"metric_best_epoch_{self.best_epoch + 1}",
            f"metric_final_train_acc_{final_metrics['train_acc']:.2f}",
            f"metric_final_eval_acc_{final_metrics['eval_acc']:.2f}",
            f"metric_final_eval_top5_{final_metrics['eval_acc_top5']:.2f}",
        ]
        for name in marker_names:
            (self.run_log_dir / name).touch(exist_ok=True)

    def _write_training_summary(self, final_metrics: dict) -> None:
        """Persist a human-readable training summary."""
        finished_at = datetime.now()
        elapsed_sec = int((finished_at - self.run_started_at).total_seconds())

        lines = [
            "Training Summary",
            "=" * 40,
            f"mode: {self.mode}",
            f"device: {self.device}",
            f"epochs_completed: {self.epochs - self.start_epoch}",
            f"best_eval_acc: {self.best_acc:.2f}",
            f"best_epoch: {self.best_epoch + 1}",
            f"final_train_acc: {final_metrics['train_acc']:.2f}",
            f"final_eval_acc: {final_metrics['eval_acc']:.2f}",
            f"final_eval_top5: {final_metrics['eval_acc_top5']:.2f}",
            f"final_train_loss: {final_metrics['train_loss']:.6f}",
            f"final_eval_loss: {final_metrics['eval_loss']:.6f}",
            f"started_at: {self.run_started_at.isoformat(timespec='seconds')}",
            f"finished_at: {finished_at.isoformat(timespec='seconds')}",
            f"elapsed_seconds: {elapsed_sec}",
            f"run_log_dir: {self.run_log_dir}",
            f"metrics_csv: {self.metrics_csv_path}",
            f"metrics_jsonl: {self.metrics_jsonl_path}",
            f"config_summary: {self.config_summary_path}",
            f"config_summary_json: {self.config_summary_json_path}",
        ]
        with open(self.summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def _build_criterion(self, config: dict) -> nn.Module:
        kd_cfg = config.get("distillation", {})

        if self.mode in ("teacher_finetune", "baseline"):
            return self.hard_criterion
        elif self.mode == "distillation":
            return KDLoss(
                temperature=kd_cfg.get("temperature", 5.0),
                alpha=kd_cfg.get("alpha", 0.7),
                label_smoothing=self.label_smoothing,
            )
        elif self.mode == "distillation_at":
            # Backward compatible: at_beta is used as fallback if
            # at_beta_spatial / at_beta_temporal are not in config.
            at_beta_fallback = kd_cfg.get("at_beta", 0.05)
            return CombinedKDATLoss(
                temperature=kd_cfg.get("temperature", 5.0),
                alpha=kd_cfg.get("alpha", 0.7),
                beta_spatial=kd_cfg.get("at_beta_spatial", at_beta_fallback),
                beta_temporal=kd_cfg.get("at_beta_temporal", at_beta_fallback),
                label_smoothing=self.label_smoothing,
                teacher_keys=kd_cfg.get("teacher_keys", [3, 4, 5]),
                student_keys=kd_cfg.get("student_keys", [2, 4, 6]),
            )
        else:
            raise ValueError(f"Unknown training mode: {self.mode}")

    def _build_optimizer(self, tr_cfg: dict) -> torch.optim.Optimizer:
        opt_name = tr_cfg.get("optimizer", "sgd").lower()
        lr = tr_cfg.get("lr", 0.01)
        wd = tr_cfg.get("weight_decay", 1e-4)

        if opt_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(), lr=lr,
                momentum=tr_cfg.get("momentum", 0.9),
                weight_decay=wd,
            )
        elif opt_name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self, tr_cfg: dict):
        sched_name = tr_cfg.get("scheduler", "cosine").lower()
        if sched_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs,
            )
        elif sched_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=tr_cfg.get("step_size", 15),
                gamma=tr_cfg.get("gamma", 0.1),
            )
        elif sched_name == "cosine_warmup":
            # Linear warmup from start_factor*lr → lr over warmup_epochs,
            # then cosine annealing for the remaining epochs.
            warmup_epochs = int(tr_cfg.get("warmup_epochs", 5))
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=tr_cfg.get("warmup_start_factor", 0.1),
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.epochs - warmup_epochs),
            )
            return torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[warmup_epochs],
            )
        elif sched_name == "none":
            return None
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    def _load_checkpoint(self, path: str) -> None:
        print(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epoch = ckpt.get("epoch", 0) + 1
        self.best_acc = ckpt.get("best_acc", 0.0)
        if self.scheduler and "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_acc": self.best_acc,
        }
        if self.scheduler:
            state["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save latest
        path = self.checkpoint_dir / f"{self.mode}_latest.pth"
        torch.save(state, path)

        if is_best:
            best_path = self.checkpoint_dir / f"{self.mode}_best.pth"
            torch.save(state, best_path)

    def train(self) -> dict:
        """Run the full training loop.

        Returns:
            Dict with final metrics.
        """
        print(f"Starting training | mode={self.mode} | epochs={self.epochs} | device={self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        last_metrics = None
        for epoch in range(self.start_epoch, self.epochs):
            self.current_epoch = epoch
            train_metrics = self._train_epoch(epoch)
            eval_metrics = self._evaluate(epoch)

            # LR scheduler step
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler:
                self.scheduler.step()

            # Check best
            is_best = eval_metrics["eval_acc"] > self.best_acc
            if is_best:
                self.best_acc = eval_metrics["eval_acc"]
                self.best_epoch = epoch
            self._save_checkpoint(epoch, is_best=is_best)

            # Log
            metrics = {
                "epoch": epoch,
                **train_metrics,
                **eval_metrics,
                "lr": current_lr,
                "best_eval_acc": self.best_acc,
                "best_acc": self.best_acc,
            }
            logger.log_metrics(metrics, step=epoch)
            self._append_epoch_metrics(metrics)
            last_metrics = metrics

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                f"Eval Acc: {eval_metrics['eval_acc']:.2f}% | "
                f"Best Eval: {self.best_acc:.2f}% | "
                f"LR: {current_lr:.6f}"
            )

        if last_metrics is not None:
            self._write_training_summary(last_metrics)
            self._write_ls_markers(last_metrics)

        print(f"Training complete. Best eval accuracy: {self.best_acc:.2f}%")
        return {
            "best_acc": self.best_acc,
            "best_epoch": self.best_epoch,
            "run_log_dir": str(self.run_log_dir),
        }

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for clips, labels in pbar:
            clips = clips.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                loss, logits = self._compute_loss(clips, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * clips.size(0)

            # Accuracy from the logits already computed for the loss
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            pbar.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

        avg_loss = running_loss / total
        acc = 100.0 * correct / total
        return {"train_loss": avg_loss, "train_acc": acc}

    def _compute_loss(self, clips: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute loss depending on training mode."""
        if self.mode in ("teacher_finetune", "baseline"):
            logits = self.model(clips)
            return self.criterion(logits, labels), logits

        elif self.mode == "distillation":
            student_logits = self.model(clips)
            if self.current_epoch < self.kd_warmup_epochs:
                return self.hard_criterion(student_logits, labels), student_logits
            with torch.no_grad():
                teacher_logits = self.teacher(clips)
            return self.criterion(student_logits, teacher_logits, labels), student_logits

        elif self.mode == "distillation_at":
            student_logits = self.model(clips)
            if self.current_epoch < self.kd_warmup_epochs:
                return self.hard_criterion(student_logits, labels), student_logits
            with torch.no_grad():
                teacher_logits = self.teacher(clips)
            teacher_feats = self.teacher.get_intermediate_features()
            student_feats = self.model.get_intermediate_features()
            total, kd, at = self.criterion(
                student_logits, teacher_logits, labels,
                teacher_feats, student_feats,
            )
            return total, student_logits

        raise ValueError(f"Unknown mode: {self.mode}")

    @torch.no_grad()
    def _evaluate(self, epoch: int) -> dict:
        self.model.eval()
        correct = 0
        correct_top5 = 0
        total = 0
        running_loss = 0.0
        ce_criterion = nn.CrossEntropyLoss()

        pbar = tqdm(self.eval_loader, desc=f"Eval {epoch+1}", leave=True, file=sys.stdout)
        for clips, labels in pbar:
            clips = clips.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                logits = self.model(clips)
                loss = ce_criterion(logits, labels)

            running_loss += loss.item() * clips.size(0)
            total += labels.size(0)

            # Top-1
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

            # Top-5
            _, top5_preds = logits.topk(5, dim=1)
            correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            pbar.set_postfix(
                acc=f"{(100.0 * correct / total):.2f}",
                top5=f"{(100.0 * correct_top5 / total):.2f}",
            )

        acc = 100.0 * correct / total
        acc_top5 = 100.0 * correct_top5 / total
        avg_loss = running_loss / total

        return {"eval_acc": acc, "eval_acc_top5": acc_top5, "eval_loss": avg_loss}
