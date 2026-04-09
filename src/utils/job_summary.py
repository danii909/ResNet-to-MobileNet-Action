#!/usr/bin/env python3
"""Aggregate run-level training artifacts into a single job summary.

This utility scans a SLURM job log directory under ``experiments/logs`` and
writes:
    - job_training_summary.txt
    - job_training_summary.json

It supports both single-run jobs (``slurm-train-<jobid>``) and sequential jobs
(``slurm-train-seq-<jobid>`` with ``run-N`` subdirectories).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


_RUN_DIR_RE = re.compile(r"run-(\d+)$")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _read_kv_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key:
                out[key] = value
    except OSError:
        return {}

    return out


def _read_colon_kv_file(path: Path) -> dict[str, str]:
    """Read ``key: value`` lines from text summaries."""
    if not path.exists():
        return {}

    out: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                out[key] = value
    except OSError:
        return {}

    return out


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _infer_training_type(mode: str, config_path: str) -> str:
    mode_l = (mode or "").strip().lower()
    cfg_l = (config_path or "").strip().lower()

    if mode_l == "teacher_finetune" or "teacher" in cfg_l:
        return "teacher"
    if mode_l == "baseline" or "baseline" in cfg_l:
        return "baseline"
    if mode_l in ("distillation", "distillation_at") or "distill" in cfg_l:
        return "distillation"
    return "unknown"


def _fmt_metric(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _find_run_dirs(job_dir: Path) -> list[Path]:
    run_dirs = [
        p for p in job_dir.iterdir()
        if p.is_dir() and _RUN_DIR_RE.fullmatch(p.name)
    ]
    if not run_dirs:
        return [job_dir]

    def run_idx(path: Path) -> int:
        m = _RUN_DIR_RE.fullmatch(path.name)
        return int(m.group(1)) if m else 0

    return sorted(run_dirs, key=run_idx)


def _run_status(run_dir: Path, run_meta: dict[str, str]) -> str:
    if (run_dir / "status_SUCCESS").exists():
        return "SUCCESS"
    if (run_dir / "status_FAILED").exists():
        return "FAILED"

    exit_code = run_meta.get("exit_code")
    if exit_code is not None and exit_code != "":
        return "SUCCESS" if exit_code == "0" else "FAILED"

    if (run_dir / "training_summary.json").exists() or (run_dir / "training_summary.txt").exists():
        return "SUCCESS"

    return "UNKNOWN"


def _extract_run_summary(job_dir: Path, run_dir: Path) -> dict[str, Any]:
    run_meta = _read_kv_file(run_dir / "job_meta.txt")
    config_summary = _read_json(run_dir / "config_summary.json")
    training_summary = _read_json(run_dir / "training_summary.json")
    training_summary_txt = _read_colon_kv_file(run_dir / "training_summary.txt")

    config_path = _coalesce(
        run_meta.get("run_config"),
        run_meta.get("config"),
        training_summary.get("config_path"),
        training_summary_txt.get("config_path"),
    )

    mode = _coalesce(
        training_summary.get("mode"),
        config_summary.get("mode"),
        training_summary_txt.get("mode"),
    )
    mode = str(mode) if mode is not None else ""

    training_type = _infer_training_type(mode, str(config_path or ""))
    status = _run_status(run_dir, run_meta)

    training_cfg = config_summary.get("training", {}) if isinstance(config_summary.get("training"), dict) else {}
    model_cfg = config_summary.get("model", {}) if isinstance(config_summary.get("model"), dict) else {}
    dist_cfg = config_summary.get("distillation", {}) if isinstance(config_summary.get("distillation"), dict) else {}

    best_acc = _to_float(_coalesce(training_summary.get("best_acc"), training_summary_txt.get("best_acc")))
    final_train_acc = _to_float(_coalesce(training_summary.get("final_train_acc"), training_summary_txt.get("final_train_acc")))
    final_test_acc = _to_float(_coalesce(training_summary.get("final_test_acc"), training_summary_txt.get("final_test_acc")))
    final_test_top5 = _to_float(_coalesce(training_summary.get("final_test_top5"), training_summary_txt.get("final_test_top5")))
    best_epoch = _to_int(_coalesce(training_summary.get("best_epoch"), training_summary_txt.get("best_epoch")))
    epochs_completed = _to_int(_coalesce(training_summary.get("epochs_completed"), training_summary_txt.get("epochs_completed")))
    elapsed_seconds = _to_int(_coalesce(training_summary.get("elapsed_seconds"), training_summary_txt.get("elapsed_seconds")))

    job_root_meta = _read_kv_file(job_dir / "job_meta.txt")
    node_name = _coalesce(run_meta.get("hostname"), job_root_meta.get("hostname"), "")
    gpu_name = _coalesce(run_meta.get("gpu_name"), job_root_meta.get("gpu_name"), "")

    return {
        "run_tag": run_dir.name,
        "run_log_dir": str(run_dir),
        "status": status,
        "training_type": training_type,
        "mode": mode,
        "profile": config_summary.get("profile"),
        "config_path": config_path,
        "node": node_name,
        "gpu": gpu_name,
        "device": _coalesce(training_summary.get("device"), training_summary_txt.get("device")),
        "started_at": _coalesce(run_meta.get("started_at"), training_summary.get("started_at"), training_summary_txt.get("started_at")),
        "finished_at": _coalesce(run_meta.get("finished_at"), training_summary.get("finished_at"), training_summary_txt.get("finished_at")),
        "exit_code": _coalesce(run_meta.get("exit_code"), ""),
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "final_test_top5": final_test_top5,
        "final_train_loss": _to_float(_coalesce(training_summary.get("final_train_loss"), training_summary_txt.get("final_train_loss"))),
        "final_test_loss": _to_float(_coalesce(training_summary.get("final_test_loss"), training_summary_txt.get("final_test_loss"))),
        "epochs_completed": epochs_completed,
        "elapsed_seconds": elapsed_seconds,
        "model": {
            "type": model_cfg.get("type"),
            "width_mult": model_cfg.get("width_mult"),
            "pretrained": model_cfg.get("pretrained"),
            "freeze_backbone": model_cfg.get("freeze_backbone"),
        },
        "training": {
            "epochs": training_cfg.get("epochs"),
            "batch_size": training_cfg.get("batch_size"),
            "optimizer": training_cfg.get("optimizer"),
            "lr": training_cfg.get("lr"),
            "momentum": training_cfg.get("momentum"),
            "weight_decay": training_cfg.get("weight_decay"),
            "scheduler": training_cfg.get("scheduler"),
            "grad_clip": training_cfg.get("grad_clip"),
            "mixed_precision": training_cfg.get("mixed_precision"),
        },
        "distillation": {
            "teacher_checkpoint": dist_cfg.get("teacher_checkpoint"),
            "temperature": dist_cfg.get("temperature"),
            "alpha": dist_cfg.get("alpha"),
            "at_beta": dist_cfg.get("at_beta"),
        },
    }


def build_job_summary(job_dir: Path) -> dict[str, Any]:
    run_dirs = _find_run_dirs(job_dir)
    runs = [_extract_run_summary(job_dir, run_dir) for run_dir in run_dirs]

    root_meta = _read_kv_file(job_dir / "job_meta.txt")
    success_runs = sum(1 for r in runs if r["status"] == "SUCCESS")
    failed_runs = sum(1 for r in runs if r["status"] == "FAILED")
    unknown_runs = len(runs) - success_runs - failed_runs

    final_test_vals = [r["final_test_acc"] for r in runs if isinstance(r["final_test_acc"], float)]
    best_acc_vals = [r["best_acc"] for r in runs if isinstance(r["best_acc"], float)]

    slurm_job_id = root_meta.get("slurm_job_id")
    if not slurm_job_id:
        m = re.search(r"(\d+)$", job_dir.name)
        slurm_job_id = m.group(1) if m else ""

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "job_tag": job_dir.name,
        "slurm_job_id": slurm_job_id,
        "job_log_dir": str(job_dir),
        "job_type": "sequential" if any(_RUN_DIR_RE.fullmatch(p.name) for p in run_dirs) else "single",
        "hostname": root_meta.get("hostname", ""),
        "gpu_name": root_meta.get("gpu_name", ""),
        "partition": root_meta.get("partition", ""),
        "qos": root_meta.get("qos", ""),
        "started_at": root_meta.get("started_at", ""),
        "finished_at": root_meta.get("finished_at", ""),
        "overall_status": root_meta.get("overall_status", ""),
        "total_runs": len(runs),
        "success_runs": success_runs,
        "failed_runs": failed_runs,
        "unknown_runs": unknown_runs,
        "best_test_acc_max": max(best_acc_vals) if best_acc_vals else None,
        "avg_final_test_acc": (sum(final_test_vals) / len(final_test_vals)) if final_test_vals else None,
        "runs": runs,
    }


def _render_text(summary: dict[str, Any]) -> str:
    lines: list[str] = [
        "Job Training Summary",
        "=" * 60,
        f"job_tag: {summary.get('job_tag', '')}",
        f"slurm_job_id: {summary.get('slurm_job_id', '')}",
        f"job_type: {summary.get('job_type', '')}",
        f"job_log_dir: {summary.get('job_log_dir', '')}",
        f"generated_at: {summary.get('generated_at', '')}",
        f"started_at: {summary.get('started_at', '')}",
        f"finished_at: {summary.get('finished_at', '')}",
        f"overall_status: {summary.get('overall_status', '')}",
        f"partition: {summary.get('partition', '')}",
        f"qos: {summary.get('qos', '')}",
        f"node: {summary.get('hostname', '')}",
        f"gpu: {summary.get('gpu_name', '')}",
        "",
        f"total_runs: {summary.get('total_runs', 0)}",
        f"success_runs: {summary.get('success_runs', 0)}",
        f"failed_runs: {summary.get('failed_runs', 0)}",
        f"unknown_runs: {summary.get('unknown_runs', 0)}",
        f"best_test_acc_max: {_fmt_metric(_to_float(summary.get('best_test_acc_max')))}",
        f"avg_final_test_acc: {_fmt_metric(_to_float(summary.get('avg_final_test_acc')))}",
        "",
        "Runs",
        "-" * 60,
    ]

    for idx, run in enumerate(summary.get("runs", []), start=1):
        lines.extend([
            f"[{idx}] {run.get('run_tag', '')}",
            f"status: {run.get('status', '')}",
            f"training_type: {run.get('training_type', '')}",
            f"mode: {run.get('mode', '')}",
            f"profile: {run.get('profile', '')}",
            f"config: {run.get('config_path', '')}",
            f"node: {run.get('node', '')}",
            f"gpu: {run.get('gpu', '')}",
            f"device: {run.get('device', '')}",
            f"started_at: {run.get('started_at', '')}",
            f"finished_at: {run.get('finished_at', '')}",
            f"elapsed_seconds: {run.get('elapsed_seconds', '')}",
            f"exit_code: {run.get('exit_code', '')}",
            f"best_acc: {_fmt_metric(_to_float(run.get('best_acc')))}",
            f"best_epoch: {run.get('best_epoch', '')}",
            f"final_train_acc: {_fmt_metric(_to_float(run.get('final_train_acc')))}",
            f"final_test_acc: {_fmt_metric(_to_float(run.get('final_test_acc')))}",
            f"final_test_top5: {_fmt_metric(_to_float(run.get('final_test_top5')))}",
            f"final_train_loss: {_fmt_metric(_to_float(run.get('final_train_loss')), digits=6)}",
            f"final_test_loss: {_fmt_metric(_to_float(run.get('final_test_loss')), digits=6)}",
            f"epochs_completed: {run.get('epochs_completed', '')}",
            "",
            "[model]",
            f"type: {run.get('model', {}).get('type', '')}",
            f"width_mult: {run.get('model', {}).get('width_mult', '')}",
            f"pretrained: {run.get('model', {}).get('pretrained', '')}",
            f"freeze_backbone: {run.get('model', {}).get('freeze_backbone', '')}",
            "",
            "[training]",
            f"epochs: {run.get('training', {}).get('epochs', '')}",
            f"batch_size: {run.get('training', {}).get('batch_size', '')}",
            f"optimizer: {run.get('training', {}).get('optimizer', '')}",
            f"lr: {run.get('training', {}).get('lr', '')}",
            f"momentum: {run.get('training', {}).get('momentum', '')}",
            f"weight_decay: {run.get('training', {}).get('weight_decay', '')}",
            f"scheduler: {run.get('training', {}).get('scheduler', '')}",
            f"grad_clip: {run.get('training', {}).get('grad_clip', '')}",
            f"mixed_precision: {run.get('training', {}).get('mixed_precision', '')}",
            "",
            "[distillation]",
            f"teacher_checkpoint: {run.get('distillation', {}).get('teacher_checkpoint', '')}",
            f"temperature: {run.get('distillation', {}).get('temperature', '')}",
            f"alpha: {run.get('distillation', {}).get('alpha', '')}",
            f"at_beta: {run.get('distillation', {}).get('at_beta', '')}",
            "-" * 60,
        ])

    return "\n".join(lines).rstrip() + "\n"


def write_job_summary(job_dir: Path) -> tuple[Path, Path]:
    summary = build_job_summary(job_dir)
    txt_path = job_dir / "job_training_summary.txt"
    json_path = job_dir / "job_training_summary.json"

    txt_path.write_text(_render_text(summary), encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return txt_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an aggregate summary for a SLURM training job directory."
    )
    parser.add_argument(
        "--job-log-dir",
        required=True,
        help="Path to job log directory (e.g. experiments/logs/slurm-train-12345)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print generated file paths",
    )
    args = parser.parse_args()

    job_dir = Path(args.job_log_dir).expanduser()
    if not job_dir.exists() or not job_dir.is_dir():
        raise SystemExit(f"Invalid --job-log-dir: {job_dir}")

    txt_path, json_path = write_job_summary(job_dir)
    if not args.quiet:
        print(f"Wrote: {txt_path}")
        print(f"Wrote: {json_path}")


if __name__ == "__main__":
    main()
