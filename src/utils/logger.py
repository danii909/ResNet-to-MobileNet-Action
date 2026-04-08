"""Weights & Biases logging wrapper."""

from typing import Any

import wandb


_run = None


def init(config: dict) -> None:
    """Initialize a W&B run from the logging section of the config.
    
    Expected config keys under 'logging':
        project: W&B project name
        run_name: optional run name
        entity: optional W&B entity/team
        enabled: bool (default True)
    """
    global _run
    log_cfg = config.get("logging", {})

    if not log_cfg.get("enabled", True):
        wandb.init(mode="disabled")
        return

    _run = wandb.init(
        project=log_cfg.get("project", "kd-action-recognition"),
        name=log_cfg.get("run_name"),
        entity=log_cfg.get("entity"),
        config=config,
        reinit="finish_previous",
    )


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a dictionary of metrics to W&B."""
    wandb.log(metrics, step=step)


def log_model_info(name: str, param_count: int, size_mb: float) -> None:
    """Log model metadata as a W&B summary entry."""
    wandb.run.summary[f"{name}/param_count"] = param_count
    wandb.run.summary[f"{name}/size_mb"] = size_mb


def finish() -> None:
    """Finish the current W&B run."""
    wandb.finish()
