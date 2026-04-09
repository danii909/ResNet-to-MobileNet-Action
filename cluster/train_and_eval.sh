#!/bin/bash
# ==========================================================================
# Single SLURM job that runs teacher, baseline, and distillation sequentially
# and stores everything under one root folder:
#   experiments/logs/slurm-train-eval-<JOBID>/
#
# Layout:
#   slurm-train-eval-<JOBID>/
#     teacher/train
#     teacher/eval
#     baseline/train
#     baseline/eval
#     distillation/train
#     distillation/eval
#     pipeline_summary.txt
# ==========================================================================

#SBATCH --job-name=kd-train-eval
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-train-eval-%j.log

set -euo pipefail

CONFIGS=(
    "experiments/configs/teacher.yaml"
    "experiments/configs/baseline.yaml"
    "experiments/configs/distillation.yaml"
)

CONFIG_NAMES=("teacher" "baseline" "distillation")

default_checkpoint_for_config() {
    local cfg="$1"
    case "$cfg" in
        *teacher*) echo "experiments/checkpoints/teacher_finetune_best.pth" ;;
        *baseline*) echo "experiments/checkpoints/baseline_best.pth" ;;
        *distill*) echo "experiments/checkpoints/distillation_best.pth" ;;
        *) echo "" ;;
    esac
}

ROOT_DIR="$HOME/dl26-projects/experiments/logs/slurm-train-eval-${SLURM_JOB_ID:-local}"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
JOB_META_FILE="$ROOT_DIR/job_meta.txt"

write_pipeline_summary() {
    python3 - "$ROOT_DIR" "$SUMMARY_FILE" "$JOB_META_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_text_kv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in raw_line:
                continue
            key, value = raw_line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key:
                data[key] = value
    except OSError:
        return {}
    return data


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def fmt(value, digits: int = 2) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


root_dir = Path(sys.argv[1])
summary_file = Path(sys.argv[2])
job_meta_file = Path(sys.argv[3])

job_meta = read_text_kv(job_meta_file)
lines: list[str] = [
    "Pipeline Summary",
    "=" * 60,
    f"job_tag: {root_dir.name}",
    f"slurm_job_id: {job_meta.get('slurm_job_id', '')}",
    f"started_at: {job_meta.get('started_at', '')}",
    f"finished_at: {job_meta.get('finished_at', '')}",
    f"overall_status: {job_meta.get('overall_status', '')}",
    f"root_dir: {root_dir}",
    f"slurm_stdout: {job_meta.get('slurm_stdout', '')}",
    "",
]

for name in ("teacher", "baseline", "distillation"):
    phase_dir = root_dir / name
    train_dir = phase_dir / "train"
    eval_dir = phase_dir / "eval"
    config_summary = read_json(train_dir / "config_summary.json")
    train_summary = read_text_kv(train_dir / "training_summary.txt")
    eval_summary = read_json(eval_dir / "evaluation_summary.json")

    status_train = "SUCCESS" if (train_dir / "status_SUCCESS").exists() else ("FAILED" if (train_dir / "status_FAILED").exists() else "UNKNOWN")
    status_eval = "SUCCESS" if (eval_dir / "status_SUCCESS").exists() else ("FAILED" if (eval_dir / "status_FAILED").exists() else "UNKNOWN")

    model_cfg = config_summary.get("model", {}) if isinstance(config_summary.get("model"), dict) else {}
    data_cfg = config_summary.get("dataset", {}) if isinstance(config_summary.get("dataset"), dict) else {}
    tr_cfg = config_summary.get("training", {}) if isinstance(config_summary.get("training"), dict) else {}
    kd_cfg = config_summary.get("distillation", {}) if isinstance(config_summary.get("distillation"), dict) else {}

    lines.extend([
        f"[{name}]",
        f"train_dir: {train_dir}",
        f"eval_dir: {eval_dir}",
        f"train_status: {status_train}",
        f"eval_status: {status_eval}",
        "[config]",
        f"mode: {config_summary.get('mode', '')}",
        f"seed: {config_summary.get('seed', '')}",
        f"model.type: {model_cfg.get('type', '')}",
        f"model.width_mult: {model_cfg.get('width_mult', '')}",
        f"model.pretrained: {model_cfg.get('pretrained', '')}",
        f"model.freeze_backbone: {model_cfg.get('freeze_backbone', '')}",
        f"dataset.data_dir: {data_cfg.get('data_dir', '')}",
        f"dataset.num_classes: {data_cfg.get('num_classes', '')}",
        f"dataset.num_frames: {data_cfg.get('num_frames', '')}",
        f"dataset.crop_size: {data_cfg.get('crop_size', '')}",
        f"dataset.backend: {data_cfg.get('backend', '')}",
        f"training.epochs: {tr_cfg.get('epochs', '')}",
        f"training.batch_size: {tr_cfg.get('batch_size', '')}",
        f"training.num_workers: {tr_cfg.get('num_workers', '')}",
        f"training.optimizer: {tr_cfg.get('optimizer', '')}",
        f"training.lr: {tr_cfg.get('lr', '')}",
        f"training.momentum: {tr_cfg.get('momentum', '')}",
        f"training.weight_decay: {tr_cfg.get('weight_decay', '')}",
        f"training.scheduler: {tr_cfg.get('scheduler', '')}",
        f"training.grad_clip: {tr_cfg.get('grad_clip', '')}",
        f"training.mixed_precision: {tr_cfg.get('mixed_precision', '')}",
    ])

    if kd_cfg:
        lines.extend([
            f"distillation.teacher_checkpoint: {kd_cfg.get('teacher_checkpoint', '')}",
            f"distillation.temperature: {kd_cfg.get('temperature', '')}",
            f"distillation.alpha: {kd_cfg.get('alpha', '')}",
            f"distillation.at_beta: {kd_cfg.get('at_beta', '')}",
        ])

    lines.extend([
        "[train results]",
        f"best_acc: {train_summary.get('best_acc', 'n/a')}",
        f"best_epoch: {train_summary.get('best_epoch', 'n/a')}",
        f"final_train_acc: {train_summary.get('final_train_acc', 'n/a')}",
        f"final_test_acc: {train_summary.get('final_test_acc', 'n/a')}",
        f"final_test_top5: {train_summary.get('final_test_top5', 'n/a')}",
        f"elapsed_seconds: {train_summary.get('elapsed_seconds', 'n/a')}",
        "[eval results]",
        f"model_type: {eval_summary.get('model_type', '')}",
        f"checkpoint: {eval_summary.get('checkpoint', '')}",
        f"top1: {fmt(eval_summary.get('top1'))}",
        f"top5: {fmt(eval_summary.get('top5'))}",
        f"size_mb: {fmt(eval_summary.get('size_mb'))}",
        f"inference_avg_ms: {fmt(eval_summary.get('inference_avg_ms'))}",
        "",
    ])

summary_file.parent.mkdir(parents=True, exist_ok=True)
summary_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"[Pipeline] Summary written to: {summary_file}")
PY
}

run_training() {
    local name="$1"
    local config="$2"
    local train_dir="$3"

    mkdir -p "$train_dir"
    echo ">>> TRAINING: $name"
    echo "    Config: $config"
    echo "    Train dir: $train_dir"

    set +e
    TRAIN_LOG_DIR="$train_dir" \
    TRAIN_CONFIG_PATH="$config" \
    TRAINING_TYPE="$name" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        /shared/sifs/latest.sif \
        python -u -m src.training.train --config "$config"
    local exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ]; then
        touch "$train_dir/status_SUCCESS"
    else
        touch "$train_dir/status_FAILED"
    fi
    printf 'exit_code=%s\n' "$exit_code" > "$train_dir/status.txt"

    return "$exit_code"
}

run_evaluation() {
    local name="$1"
    local config="$2"
    local checkpoint="$3"
    local eval_dir="$4"

    mkdir -p "$eval_dir"
    echo ">>> EVALUATING: $name"
    echo "    Config: $config"
    echo "    Checkpoint: $checkpoint"
    echo "    Eval dir: $eval_dir"

    set +e
    EVAL_LOG_DIR="$eval_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.evaluate --config "$config" --override "evaluation.checkpoint=$checkpoint"
    local exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ]; then
        touch "$eval_dir/status_SUCCESS"
    else
        touch "$eval_dir/status_FAILED"
    fi
    printf 'exit_code=%s\n' "$exit_code" > "$eval_dir/status.txt"

    return "$exit_code"
}

mkdir -p "$ROOT_DIR" logs

STARTED_AT="$(date --iso-8601=seconds)"
{
    echo "job_tag=$(basename "$ROOT_DIR")"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "hostname=$(hostname)"
    echo "node=$(hostname)"
    echo "started_at=$STARTED_AT"
    echo "project_dir=$HOME/dl26-projects"
    echo "configs=${CONFIGS[*]}"
    echo "overall_status=RUNNING"
} > "$JOB_META_FILE"

echo "=========================================="
echo "TRAIN & EVAL PIPELINE - single SLURM job"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Root dir: $ROOT_DIR"
echo "Starting at: $STARTED_AT"
echo "Teacher      (40 epochs) + eval"
echo "Baseline     (60 epochs) + eval"
echo "Distillation (60 epochs) + eval"
echo ""

for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    name="${CONFIG_NAMES[$i]}"
    phase_dir="$ROOT_DIR/$name"
    train_dir="$phase_dir/train"
    eval_dir="$phase_dir/eval"

    mkdir -p "$train_dir" "$eval_dir"

    run_training "$name" "$config" "$train_dir"

    checkpoint="$(default_checkpoint_for_config "$config")"
    if [ -z "$checkpoint" ]; then
        echo "ERROR: can't infer checkpoint for $config"
        touch "$train_dir/status_FAILED"
        exit 1
    fi

    run_evaluation "$name" "$config" "$checkpoint" "$eval_dir"
    echo ""
done

FINISHED_AT="$(date --iso-8601=seconds)"
{
    echo "job_tag=$(basename "$ROOT_DIR")"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "hostname=$(hostname)"
    echo "node=$(hostname)"
    echo "started_at=$STARTED_AT"
    echo "finished_at=$FINISHED_AT"
    echo "project_dir=$HOME/dl26-projects"
    echo "configs=${CONFIGS[*]}"
    echo "overall_status=COMPLETED"
    echo "slurm_stdout=${SLURM_SUBMIT_DIR:-$HOME/dl26-projects}/logs/slurm-train-eval-${SLURM_JOB_ID:-local}.log"
} > "$JOB_META_FILE"

write_pipeline_summary

if [ -f "${SLURM_SUBMIT_DIR:-$HOME/dl26-projects}/logs/slurm-train-eval-${SLURM_JOB_ID:-local}.log" ]; then
    cp "${SLURM_SUBMIT_DIR:-$HOME/dl26-projects}/logs/slurm-train-eval-${SLURM_JOB_ID:-local}.log" "$ROOT_DIR/slurm-stdout.log" || true
fi

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "Finished at: $FINISHED_AT"
echo "Root dir: $ROOT_DIR"
echo "=========================================="
