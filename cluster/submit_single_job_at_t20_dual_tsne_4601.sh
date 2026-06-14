#!/bin/bash
# ============================================================================
# Single SLURM job: two AT runs (symmetric + temporal, T=20).
# For each run:
#   1) train student with the provided AT config
#   2) evaluate on the official test split
#   3) generate a t-SNE comparison: teacher vs baseline vs AT student
# Final artifacts are mirrored under results/Train-eval-test-split (group-aware)/
# Training/slurm-train-eval-<JOBID>/ so the output layout matches the 4601-style
# folder you referenced.
# ============================================================================

#SBATCH --job-name=at-t20-dual-4601
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

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi
cd "$PROJECT_DIR"
mkdir -p logs

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-train-eval-${SLURM_JOB_ID:-local}"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-${SLURM_JOB_ID:-local}"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
TSNE_DIR="$ROOT_DIR/tsne_plots"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT" "$TSNE_DIR"

AT_SYM_CFG="experiments/configs/valid/at_symmetric_t20.yaml"
AT_TEMP_CFG="experiments/configs/valid/at_temporal_t20.yaml"
BASELINE_CKPT="${BASELINE_CKPT:-$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth}"

overall_status="SUCCESS"

extract_teacher_ckpt() {
    local config="$1"
    local teacher_line
    teacher_line="$(grep -E '^[[:space:]]*teacher_checkpoint:[[:space:]]*' "$config" | head -n 1 || true)"
    if [ -z "$teacher_line" ]; then
        echo "ERROR: teacher_checkpoint not found in $config"
        exit 1
    fi
    echo "$teacher_line" | sed -E 's/^[[:space:]]*teacher_checkpoint:[[:space:]]*//' | tr -d '\r'
}

run_training() {
    local phase="$1"
    local config="$2"
    local train_dir="$3"
    local overrides="$4"

    mkdir -p "$train_dir"

    local -a ov=()
    if [ -n "$overrides" ]; then
        read -r -a ov <<< "$overrides"
    fi

    echo ""
    echo ">>> TRAINING: $phase"
    echo "    config: $config"

    set +e
    TRAIN_LOG_DIR="$train_dir" \
    TRAIN_CONFIG_PATH="$config" \
    TRAINING_TYPE="$phase" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.training.train --config "$config" --override "${ov[@]}"
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        touch "$train_dir/status_SUCCESS"
    else
        touch "$train_dir/status_FAILED"
        overall_status="FAILED"
    fi
    printf 'exit_code=%s\n' "$rc" > "$train_dir/status.txt"

    return "$rc"
}

run_evaluation() {
    local phase="$1"
    local config="$2"
    local checkpoint="$3"
    local eval_dir="$4"

    mkdir -p "$eval_dir"

    echo ""
    echo ">>> EVALUATING (test split): $phase"
    echo "    checkpoint: $checkpoint"

    set +e
    EVAL_LOG_DIR="$eval_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.evaluate \
            --config "$config" \
            --override "evaluation.checkpoint=$checkpoint"
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        touch "$eval_dir/status_SUCCESS"
    else
        touch "$eval_dir/status_FAILED"
        overall_status="FAILED"
    fi
    printf 'exit_code=%s\n' "$rc" > "$eval_dir/status.txt"

    return "$rc"
}

run_tsne() {
    local phase="$1"
    local config="$2"
    local student_ckpt="$3"
    local output_name="$4"

    if [ ! -f "$TEACHER_CKPT" ]; then
        echo "WARNING: missing teacher checkpoint: $TEACHER_CKPT"
        return 1
    fi
    if [ ! -f "$BASELINE_CKPT" ]; then
        echo "WARNING: missing baseline checkpoint: $BASELINE_CKPT"
        return 1
    fi
    if [ ! -f "$student_ckpt" ]; then
        echo "WARNING: missing student checkpoint for $phase: $student_ckpt"
        return 1
    fi

    echo ""
    echo ">>> TSNE: $phase"

    set +e
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.tsne_visualizer \
            --config "$config" \
            --teacher-ckpt "$TEACHER_CKPT" \
            --baseline-ckpt "$BASELINE_CKPT" \
            --student-ckpt "$student_ckpt" \
            --num-classes 10 \
            --output-dir "$TSNE_DIR" \
            --output-filename "$output_name"
    local rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        overall_status="FAILED"
    fi

    return "$rc"
}

append_summary() {
    local phase="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local checkpoint="$4"
    local train_status="$5"
    local eval_status="$6"
    local tsne_name="$7"

    python3 - "$phase" "$train_dir" "$eval_dir" "$checkpoint" "$train_status" "$eval_status" "$tsne_name" >> "$SUMMARY_FILE" <<'PY'
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


def read_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        out[k.strip()] = v.strip()
    return out


phase, train_dir, eval_dir, checkpoint, train_status, eval_status, tsne_name = sys.argv[1:]
train_dir = Path(train_dir)
eval_dir = Path(eval_dir)

cfg = read_json(train_dir / "config_summary.json")
train = read_kv(train_dir / "training_summary.txt")
ev = read_json(eval_dir / "evaluation_summary.json")

print(f"[{phase}]")
print(f"train_dir: {train_dir}")
print(f"eval_dir: {eval_dir}")
print(f"train_status: {train_status}")
print(f"eval_status: {eval_status}")
print("[config]")
print(f"mode: {cfg.get('mode', '')}")
print(f"model.type: {cfg.get('model', {}).get('type', '') if isinstance(cfg.get('model'), dict) else ''}")
print(f"training.epochs: {cfg.get('training', {}).get('epochs', '') if isinstance(cfg.get('training'), dict) else ''}")
print(f"training.batch_size: {cfg.get('training', {}).get('batch_size', '') if isinstance(cfg.get('training'), dict) else ''}")
print(f"training.lr: {cfg.get('training', {}).get('lr', '') if isinstance(cfg.get('training'), dict) else ''}")
print(f"training.scheduler: {cfg.get('training', {}).get('scheduler', '') if isinstance(cfg.get('training'), dict) else ''}")
print(f"distillation.teacher_checkpoint: {cfg.get('distillation', {}).get('teacher_checkpoint', '') if isinstance(cfg.get('distillation'), dict) else ''}")
print(f"distillation.temperature: {cfg.get('distillation', {}).get('temperature', '') if isinstance(cfg.get('distillation'), dict) else ''}")
print(f"distillation.alpha: {cfg.get('distillation', {}).get('alpha', '') if isinstance(cfg.get('distillation'), dict) else ''}")
print(f"distillation.at_beta_spatial: {cfg.get('distillation', {}).get('at_beta_spatial', '') if isinstance(cfg.get('distillation'), dict) else ''}")
print(f"distillation.at_beta_temporal: {cfg.get('distillation', {}).get('at_beta_temporal', '') if isinstance(cfg.get('distillation'), dict) else ''}")
print("[train results]")
print(f"best_eval_acc: {train.get('best_eval_acc', train.get('best_acc', 'n/a'))}")
print(f"best_epoch: {train.get('best_epoch', 'n/a')}")
print(f"final_train_acc: {train.get('final_train_acc', 'n/a')}")
print(f"final_eval_acc: {train.get('final_eval_acc', train.get('final_test_acc', 'n/a'))}")
print(f"final_eval_top5: {train.get('final_eval_top5', train.get('final_test_top5', 'n/a'))}")
print("[eval results]")
print(f"checkpoint: {ev.get('checkpoint', checkpoint)}")
print(f"top1: {ev.get('top1', 'n/a')}")
print(f"top5: {ev.get('top5', 'n/a')}")
print(f"tsne_plot: {tsne_name}.png")
print("")
PY
}

TEACHER_CKPT="$(extract_teacher_ckpt "$AT_SYM_CFG")"

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
    exit 1
fi

if [ ! -f "$BASELINE_CKPT" ]; then
    echo "ERROR: Baseline checkpoint not found: $BASELINE_CKPT"
    exit 1
fi

echo "Teacher checkpoint: $TEACHER_CKPT"
echo "Baseline checkpoint: $BASELINE_CKPT"

{
    echo "AT T20 Dual-Stage Pipeline"
    echo "============================================================"
    echo "job_id: ${SLURM_JOB_ID:-local}"
    echo "root_dir: $ROOT_DIR"
    echo "teacher_checkpoint: $TEACHER_CKPT"
    echo "baseline_checkpoint: $BASELINE_CKPT"
    echo ""
} > "$SUMMARY_FILE"

# ---------------------------------------------------------------------------
# Stage 1: AT symmetric (T=20)
# ---------------------------------------------------------------------------
STAGE_SYM="at_symmetric"
SYM_TRAIN_DIR="$ROOT_DIR/$STAGE_SYM/train"
SYM_EVAL_DIR="$ROOT_DIR/$STAGE_SYM/eval"
SYM_CKPT_DIR="$CHECKPOINT_ROOT/$STAGE_SYM"
SYM_TSNE_NAME="tsne_teacher_baseline_distill_at_symmetric"

run_training "$STAGE_SYM" "$AT_SYM_CFG" "$SYM_TRAIN_DIR" \
    "training.checkpoint_dir=$SYM_CKPT_DIR distillation.teacher_checkpoint=$TEACHER_CKPT logging.run_name=$STAGE_SYM"

SYM_CKPT="$SYM_CKPT_DIR/distillation_at_best.pth"
if [ -f "$SYM_CKPT" ]; then
    run_evaluation "$STAGE_SYM" "$AT_SYM_CFG" "$SYM_CKPT" "$SYM_EVAL_DIR"
    run_tsne "$STAGE_SYM" "$AT_SYM_CFG" "$SYM_CKPT" "$SYM_TSNE_NAME" || true
else
    overall_status="FAILED"
    mkdir -p "$SYM_EVAL_DIR"
    touch "$SYM_EVAL_DIR/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$SYM_EVAL_DIR/status.txt"
fi

SYM_TRAIN_STATUS="UNKNOWN"
SYM_EVAL_STATUS="UNKNOWN"
[ -f "$SYM_TRAIN_DIR/status_SUCCESS" ] && SYM_TRAIN_STATUS="SUCCESS"
[ -f "$SYM_TRAIN_DIR/status_FAILED" ] && SYM_TRAIN_STATUS="FAILED"
[ -f "$SYM_EVAL_DIR/status_SUCCESS" ] && SYM_EVAL_STATUS="SUCCESS"
[ -f "$SYM_EVAL_DIR/status_FAILED" ] && SYM_EVAL_STATUS="FAILED"

append_summary "$STAGE_SYM" "$SYM_TRAIN_DIR" "$SYM_EVAL_DIR" "$SYM_CKPT" "$SYM_TRAIN_STATUS" "$SYM_EVAL_STATUS" "$SYM_TSNE_NAME"

# ---------------------------------------------------------------------------
# Stage 2: AT temporal-only (T=20)
# ---------------------------------------------------------------------------
STAGE_TEMP="at_temporal_only"
TEMP_TRAIN_DIR="$ROOT_DIR/$STAGE_TEMP/train"
TEMP_EVAL_DIR="$ROOT_DIR/$STAGE_TEMP/eval"
TEMP_CKPT_DIR="$CHECKPOINT_ROOT/$STAGE_TEMP"
TEMP_TSNE_NAME="tsne_teacher_baseline_distill_at_temporal"

run_training "$STAGE_TEMP" "$AT_TEMP_CFG" "$TEMP_TRAIN_DIR" \
    "training.checkpoint_dir=$TEMP_CKPT_DIR distillation.teacher_checkpoint=$TEACHER_CKPT logging.run_name=$STAGE_TEMP"

TEMP_CKPT="$TEMP_CKPT_DIR/distillation_at_best.pth"
if [ -f "$TEMP_CKPT" ]; then
    run_evaluation "$STAGE_TEMP" "$AT_TEMP_CFG" "$TEMP_CKPT" "$TEMP_EVAL_DIR"
    run_tsne "$STAGE_TEMP" "$AT_TEMP_CFG" "$TEMP_CKPT" "$TEMP_TSNE_NAME" || true
else
    overall_status="FAILED"
    mkdir -p "$TEMP_EVAL_DIR"
    touch "$TEMP_EVAL_DIR/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$TEMP_EVAL_DIR/status.txt"
fi

TEMP_TRAIN_STATUS="UNKNOWN"
TEMP_EVAL_STATUS="UNKNOWN"
[ -f "$TEMP_TRAIN_DIR/status_SUCCESS" ] && TEMP_TRAIN_STATUS="SUCCESS"
[ -f "$TEMP_TRAIN_DIR/status_FAILED" ] && TEMP_TRAIN_STATUS="FAILED"
[ -f "$TEMP_EVAL_DIR/status_SUCCESS" ] && TEMP_EVAL_STATUS="SUCCESS"
[ -f "$TEMP_EVAL_DIR/status_FAILED" ] && TEMP_EVAL_STATUS="FAILED"

append_summary "$STAGE_TEMP" "$TEMP_TRAIN_DIR" "$TEMP_EVAL_DIR" "$TEMP_CKPT" "$TEMP_TRAIN_STATUS" "$TEMP_EVAL_STATUS" "$TEMP_TSNE_NAME"

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
echo ""
echo "Pipeline completed. Summary: $SUMMARY_FILE"
echo "t-SNE directory: $TSNE_DIR"

if [ "$overall_status" != "SUCCESS" ]; then
    exit 1
fi

echo "Overall status: SUCCESS"
