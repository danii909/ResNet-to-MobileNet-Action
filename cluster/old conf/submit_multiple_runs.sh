#!/bin/bash
# ============================================================================
# Single SLURM job for next KD runs (no teacher retrain), all sequential.
#
# Usage:
#   sbatch cluster/submit_multiple_runs.sh
#
# Constraint handled: cluster accepts one job at a time.
# ============================================================================

#SBATCH --job-name=kd-next-4041
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-multiple-runs-%j.log

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
JOB4041_ROOT="${JOB4041_ROOT:-/home/brbdnl01e03e017o/dl26-projects/experiments/logs/slurm-train-eval-4041}"
TEACHER_CKPT="${TEACHER_CKPT:-/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs

ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-multiple-runs-${SLURM_JOB_ID:-local}"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-multiple-runs-${SLURM_JOB_ID:-local}"
mkdir -p "$ROOT_DIR"
mkdir -p "$CHECKPOINT_ROOT"

echo "============================================"
echo "Single-job KD next runs (minimum goals)"
echo "Job ID:            ${SLURM_JOB_ID:-local}"
echo "Project dir:       $PROJECT_DIR"
echo "Reference job:     $JOB4041_ROOT"
echo "Teacher checkpoint: $TEACHER_CKPT"
echo "Root log dir:      $ROOT_DIR"
echo "============================================"

# Guard rails to ensure we are reusing the intended teacher from job 4041
if [ ! -f "$JOB4041_ROOT/pipeline_summary.txt" ]; then
    echo "Missing pipeline summary: $JOB4041_ROOT/pipeline_summary.txt"
    exit 1
fi

if ! grep -q "\[teacher\]" "$JOB4041_ROOT/pipeline_summary.txt"; then
    echo "Teacher block not found in pipeline summary."
    exit 1
fi

if ! grep -q "train_status: SUCCESS" "$JOB4041_ROOT/pipeline_summary.txt"; then
    echo "Teacher train status SUCCESS not found in pipeline summary."
    exit 1
fi

if ! grep -q "eval_status: SUCCESS" "$JOB4041_ROOT/pipeline_summary.txt"; then
    echo "Teacher eval status SUCCESS not found in pipeline summary."
    exit 1
fi

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Teacher checkpoint not found: $TEACHER_CKPT"
    exit 1
fi

CKPT_SIZE_MB=$(du -m "$TEACHER_CKPT" | awk '{print $1}')
if [ "${CKPT_SIZE_MB:-0}" -lt 100 ]; then
    echo "Teacher checkpoint seems too small (${CKPT_SIZE_MB} MB): $TEACHER_CKPT"
    exit 1
fi

run_training() {
    local run_name="$1"
    local config="$2"
    local train_dir="$3"
    local train_overrides="$4"

    mkdir -p "$train_dir"
    echo ""
    echo ">>> TRAINING: $run_name"
    echo "    config: $config"
    echo "    train_dir: $train_dir"

    local -a ov=()
    if [ -n "$train_overrides" ]; then
        read -r -a ov <<< "$train_overrides"
    fi

    set +e
    if [ "${#ov[@]}" -gt 0 ]; then
        TRAIN_LOG_DIR="$train_dir" \
        TRAIN_CONFIG_PATH="$config" \
        TRAINING_TYPE="$run_name" \
        apptainer run --nv \
            --env WANDB_MODE=offline \
            --env HF_DATASETS_OFFLINE=1 \
            ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
            --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
            --env PYTHONUNBUFFERED=1 \
            /shared/sifs/latest.sif \
            python -u -m src.training.train --config "$config" --override "${ov[@]}"
    else
        TRAIN_LOG_DIR="$train_dir" \
        TRAIN_CONFIG_PATH="$config" \
        TRAINING_TYPE="$run_name" \
        apptainer run --nv \
            --env WANDB_MODE=offline \
            --env HF_DATASETS_OFFLINE=1 \
            ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
            --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
            --env PYTHONUNBUFFERED=1 \
            /shared/sifs/latest.sif \
            python -u -m src.training.train --config "$config"
    fi
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
    local run_name="$1"
    local config="$2"
    local checkpoint="$3"
    local eval_dir="$4"
    local eval_overrides="$5"

    mkdir -p "$eval_dir"
    echo ""
    echo ">>> EVALUATING: $run_name"
    echo "    config: $config"
    echo "    checkpoint: $checkpoint"
    echo "    eval_dir: $eval_dir"

    local -a ov=("evaluation.checkpoint=$checkpoint")
    if [ -n "$eval_overrides" ]; then
        local -a extra=()
        read -r -a extra <<< "$eval_overrides"
        ov+=("${extra[@]}")
    fi

    set +e
    EVAL_LOG_DIR="$eval_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.evaluate --config "$config" --override "${ov[@]}"
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

append_summary() {
    local run_name="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local checkpoint="$4"

    local train_status="UNKNOWN"
    local eval_status="UNKNOWN"
    [ -f "$train_dir/status_SUCCESS" ] && train_status="SUCCESS"
    [ -f "$train_dir/status_FAILED" ] && train_status="FAILED"
    [ -f "$eval_dir/status_SUCCESS" ] && eval_status="SUCCESS"
    [ -f "$eval_dir/status_FAILED" ] && eval_status="FAILED"

    python3 - "$run_name" "$train_dir" "$eval_dir" "$checkpoint" "$train_status" "$eval_status" >> "$SUMMARY_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_colon_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or ":" not in line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    except Exception:
        return {}
    return out


run_name = sys.argv[1]
train_dir = Path(sys.argv[2])
eval_dir = Path(sys.argv[3])
checkpoint = sys.argv[4]
train_status = sys.argv[5]
eval_status = sys.argv[6]

cfg = read_json(train_dir / "config_summary.json")
train = read_colon_kv(train_dir / "training_summary.txt")
ev = read_json(eval_dir / "evaluation_summary.json")

model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
data_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
tr_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
kd_cfg = cfg.get("distillation", {}) if isinstance(cfg.get("distillation"), dict) else {}

print(f"[{run_name}]")
print(f"train_dir: {train_dir}")
print(f"eval_dir: {eval_dir}")
print(f"train_status: {train_status}")
print(f"eval_status: {eval_status}")
print("[config]")
print(f"mode: {cfg.get('mode', '')}")
print(f"seed: {cfg.get('seed', '')}")
print(f"model.type: {model_cfg.get('type', '')}")
print(f"model.width_mult: {model_cfg.get('width_mult', '')}")
print(f"model.pretrained: {model_cfg.get('pretrained', '')}")
print(f"model.freeze_backbone: {model_cfg.get('freeze_backbone', '')}")
print(f"dataset.data_dir: {data_cfg.get('data_dir', '')}")
print(f"dataset.num_classes: {data_cfg.get('num_classes', '')}")
print(f"dataset.num_frames: {data_cfg.get('num_frames', '')}")
print(f"dataset.crop_size: {data_cfg.get('crop_size', '')}")
print(f"dataset.backend: {data_cfg.get('backend', '')}")
print(f"training.epochs: {tr_cfg.get('epochs', '')}")
print(f"training.batch_size: {tr_cfg.get('batch_size', '')}")
print(f"training.num_workers: {tr_cfg.get('num_workers', '')}")
print(f"training.optimizer: {tr_cfg.get('optimizer', '')}")
print(f"training.lr: {tr_cfg.get('lr', '')}")
print(f"training.momentum: {tr_cfg.get('momentum', '')}")
print(f"training.weight_decay: {tr_cfg.get('weight_decay', '')}")
print(f"training.scheduler: {tr_cfg.get('scheduler', '')}")
print(f"training.grad_clip: {tr_cfg.get('grad_clip', '')}")
print(f"training.mixed_precision: {tr_cfg.get('mixed_precision', '')}")
print(f"training.label_smoothing: {tr_cfg.get('label_smoothing', '')}")
print(f"training.kd_warmup_epochs: {tr_cfg.get('kd_warmup_epochs', '')}")
print(f"distillation.teacher_checkpoint: {kd_cfg.get('teacher_checkpoint', '')}")
print(f"distillation.temperature: {kd_cfg.get('temperature', '')}")
print(f"distillation.alpha: {kd_cfg.get('alpha', '')}")
print(f"distillation.at_beta: {kd_cfg.get('at_beta', '')}")
print("[train results]")
print(f"best_acc: {train.get('best_acc', 'n/a')}")
print(f"best_epoch: {train.get('best_epoch', 'n/a')}")
print(f"final_train_acc: {train.get('final_train_acc', 'n/a')}")
print(f"final_test_acc: {train.get('final_test_acc', 'n/a')}")
print(f"final_test_top5: {train.get('final_test_top5', 'n/a')}")
print(f"elapsed_seconds: {train.get('elapsed_seconds', 'n/a')}")
print("[eval results]")
print(f"model_type: {ev.get('model_type', '')}")
print(f"checkpoint: {ev.get('checkpoint', checkpoint)}")
print(f"top1: {ev.get('top1', 'n/a')}")
print(f"top5: {ev.get('top5', 'n/a')}")
print(f"size_mb: {ev.get('size_mb', 'n/a')}")
print(f"inference_avg_ms: {ev.get('inference_avg_ms', 'n/a')}")
print("")
PY
}

{
    echo "Pipeline Summary"
    echo "============================================================"
    echo "job_tag: slurm-multiple-runs-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo "teacher_source_job: $JOB4041_ROOT"
    echo "teacher_checkpoint: $TEACHER_CKPT"
    echo ""
} > "$SUMMARY_FILE"

# Run A: baseline regularized
RUN="baseline_wd002_ls01"
TRAIN_DIR="$ROOT_DIR/$RUN/train"
EVAL_DIR="$ROOT_DIR/$RUN/eval"
run_training "$RUN" "experiments/configs/baseline.yaml" "$TRAIN_DIR" \
    "training.weight_decay=0.02 training.checkpoint_dir=$CHECKPOINT_ROOT/$RUN logging.run_name=baseline-wd002-ls01"
run_evaluation "$RUN" "experiments/configs/baseline.yaml" \
    "$CHECKPOINT_ROOT/$RUN/baseline_best.pth" "$EVAL_DIR" ""
append_summary "$RUN" "$TRAIN_DIR" "$EVAL_DIR" "$CHECKPOINT_ROOT/$RUN/baseline_best.pth"

# Run B: KD T6 a0.6
RUN="kd_t6_a06"
TRAIN_DIR="$ROOT_DIR/$RUN/train"
EVAL_DIR="$ROOT_DIR/$RUN/eval"
run_training "$RUN" "experiments/configs/distillation.yaml" "$TRAIN_DIR" \
    "distillation.teacher_checkpoint=$TEACHER_CKPT distillation.temperature=6.0 distillation.alpha=0.6 training.lr=0.0006 training.weight_decay=0.01 training.kd_warmup_epochs=5 training.checkpoint_dir=$CHECKPOINT_ROOT/$RUN logging.run_name=kd-t6-a06"
run_evaluation "$RUN" "experiments/configs/distillation.yaml" \
    "$CHECKPOINT_ROOT/$RUN/distillation_best.pth" "$EVAL_DIR" ""
append_summary "$RUN" "$TRAIN_DIR" "$EVAL_DIR" "$CHECKPOINT_ROOT/$RUN/distillation_best.pth"

# Run C: KD T10 a0.7
RUN="kd_t10_a07"
TRAIN_DIR="$ROOT_DIR/$RUN/train"
EVAL_DIR="$ROOT_DIR/$RUN/eval"
run_training "$RUN" "experiments/configs/distillation.yaml" "$TRAIN_DIR" \
    "distillation.teacher_checkpoint=$TEACHER_CKPT distillation.temperature=10.0 distillation.alpha=0.7 training.lr=0.0006 training.weight_decay=0.01 training.kd_warmup_epochs=5 training.checkpoint_dir=$CHECKPOINT_ROOT/$RUN logging.run_name=kd-t10-a07"
run_evaluation "$RUN" "experiments/configs/distillation.yaml" \
    "$CHECKPOINT_ROOT/$RUN/distillation_best.pth" "$EVAL_DIR" ""
append_summary "$RUN" "$TRAIN_DIR" "$EVAL_DIR" "$CHECKPOINT_ROOT/$RUN/distillation_best.pth"

# Run D: KD T6 a0.6 LR 5e-4 WD 0.02
RUN="kd_t6_a06_lr5e4_wd002"
TRAIN_DIR="$ROOT_DIR/$RUN/train"
EVAL_DIR="$ROOT_DIR/$RUN/eval"
run_training "$RUN" "experiments/configs/distillation.yaml" "$TRAIN_DIR" \
    "distillation.teacher_checkpoint=$TEACHER_CKPT distillation.temperature=6.0 distillation.alpha=0.6 training.lr=0.0005 training.weight_decay=0.02 training.kd_warmup_epochs=5 training.checkpoint_dir=$CHECKPOINT_ROOT/$RUN logging.run_name=kd-t6-a06-lr5e4-wd002"
run_evaluation "$RUN" "experiments/configs/distillation.yaml" \
    "$CHECKPOINT_ROOT/$RUN/distillation_best.pth" "$EVAL_DIR" ""
append_summary "$RUN" "$TRAIN_DIR" "$EVAL_DIR" "$CHECKPOINT_ROOT/$RUN/distillation_best.pth"

echo ""
echo "All runs completed in this single SLURM job."
echo "Summary: $SUMMARY_FILE"
