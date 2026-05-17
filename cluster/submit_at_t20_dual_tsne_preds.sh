#!/bin/bash
# ============================================================================
# Single SLURM job: train AT symmetric + AT temporal (T=20), 4511-style logs.
# Then run split t-SNE plots and save predictions CSVs.
# ============================================================================

#SBATCH --job-name=at-t20-dual
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
PRED_DIR="$ROOT_DIR/predictions"
TSNE_DIR="$ROOT_DIR/tsne_plots"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT" "$PRED_DIR" "$TSNE_DIR"

TEACHER_CKPT="${TEACHER_CKPT:-$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth}"
BASELINE_CKPT="${BASELINE_CKPT:-$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth}"
DISTILL_CKPT="${DISTILL_CKPT:-$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4565/kd_t20_a07_24f_lightaug/distillation_best.pth}"

AT_SYM_CFG="experiments/configs/valid/at_symmetric_t20.yaml"
AT_TEMP_CFG="experiments/configs/valid/at_temporal_t20.yaml"
TSNE_CFG="$AT_SYM_CFG"
TEACHER_CONFIG="experiments/configs/teacher_24f_evalsplit.yaml"
STUDENT_CONFIG="$AT_SYM_CFG"

overall_status="SUCCESS"

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
    echo ">>> EVALUATING: $phase"
    echo "    checkpoint: $checkpoint"

    set +e
    EVAL_LOG_DIR="$eval_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
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

append_summary() {
    local stage="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local ckpt="$4"
    local train_status="$5"
    local eval_status="$6"

    python3 - "$stage" "$train_dir" "$eval_dir" "$ckpt" "$train_status" "$eval_status" >> "$SUMMARY_FILE" <<'PY'
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

def read_kv(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw:
            continue
        k, v = raw.split(":", 1)
        out[k.strip()] = v.strip()
    return out

stage, train_dir, eval_dir, ckpt, train_status, eval_status = sys.argv[1:]
train_dir = Path(train_dir)
eval_dir = Path(eval_dir)

cfg = read_json(train_dir / "config_summary.json")
train = read_kv(train_dir / "training_summary.txt")
ev = read_json(eval_dir / "evaluation_summary.json")

print(f"[{stage}]")
print(f"train_status: {train_status}")
print(f"best_eval_acc: {train.get('best_eval_acc', train.get('best_acc', 'n/a'))}")
print(f"final_train_acc: {train.get('final_train_acc', 'n/a')}")
print(f"test_top1: {ev.get('top1', 'n/a')}")
print(f"teacher: {cfg.get('distillation', {}).get('teacher_checkpoint', 'n/a')}")
print("")
PY
}

pick_best_ckpt() {
    local dir="$1"
    if [ -f "$dir/distillation_at_best.pth" ]; then
        echo "$dir/distillation_at_best.pth"
        return 0
    fi
    if [ -f "$dir/distillation_best.pth" ]; then
        echo "$dir/distillation_best.pth"
        return 0
    fi
    echo ""
    return 1
}

run_predictions() {
    local label="$1"
    local model_type="$2"
    local config="$3"
    local checkpoint="$4"

    if [ ! -f "$checkpoint" ]; then
        echo "WARNING: missing checkpoint for $label: $checkpoint"
        return 1
    fi

    echo ""
    echo ">>> PREDICTIONS: $label"

    set +e
    PREDICTIONS_LOG_DIR="$PRED_DIR" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.confusion \
            --config "$config" \
            --override \
                "model.type=$model_type" \
                "evaluation.checkpoint=$checkpoint" \
                "evaluation.label=$label" \
                "evaluation.save_predictions_only=true" \
                "evaluation.predictions_dir=$PRED_DIR"
    local rc=$?
    set -e

    return "$rc"
}

run_tsne() {
    local output_name="$1"
    local student2_ckpt="$2"

    if [ ! -f "$TEACHER_CKPT" ]; then
        echo "WARNING: missing teacher checkpoint: $TEACHER_CKPT"
        return 1
    fi
    if [ ! -f "$BASELINE_CKPT" ]; then
        echo "WARNING: missing baseline checkpoint: $BASELINE_CKPT"
        return 1
    fi
    if [ ! -f "$DISTILL_CKPT" ]; then
        echo "WARNING: missing distillation checkpoint: $DISTILL_CKPT"
        return 1
    fi
    if [ ! -f "$student2_ckpt" ]; then
        echo "WARNING: missing student2 checkpoint: $student2_ckpt"
        return 1
    fi

    echo ""
    echo ">>> TSNE: $output_name"

    set +e
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env TMPDIR="$TMPDIR" \
        --env PYTHONUNBUFFERED=1 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.tsne_visualizer \
            --config "$TSNE_CFG" \
            --teacher-ckpt "$TEACHER_CKPT" \
            --baseline-ckpt "$BASELINE_CKPT" \
            --student-ckpt "$DISTILL_CKPT" \
            --student2-ckpt "$student2_ckpt" \
            --num-classes 10 \
            --output-dir "$TSNE_DIR" \
            --output-filename "$output_name"
    local rc=$?
    set -e

    return "$rc"
}

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
    exit 1
fi

{
    echo "AT T20 Dual-Stage Pipeline"
    echo "============================================================"
    echo "job_id: ${SLURM_JOB_ID:-local}"
    echo "root_dir: $ROOT_DIR"
    echo "teacher_checkpoint: $TEACHER_CKPT"
    echo ""
} > "$SUMMARY_FILE"

# ---------------------------------------------------------------------------
# Stage 1: AT symmetric (T=20)
# ---------------------------------------------------------------------------
STAGE_SYM="at_symmetric"
SYM_TRAIN_DIR="$ROOT_DIR/$STAGE_SYM/train"
SYM_EVAL_DIR="$ROOT_DIR/$STAGE_SYM/eval"
SYM_CKPT_DIR="$CHECKPOINT_ROOT/$STAGE_SYM"

run_training "$STAGE_SYM" "$AT_SYM_CFG" "$SYM_TRAIN_DIR" \
    "training.checkpoint_dir=$SYM_CKPT_DIR distillation.teacher_checkpoint=$TEACHER_CKPT logging.run_name=$STAGE_SYM" || true

SYM_CKPT="$(pick_best_ckpt "$SYM_CKPT_DIR")"
if [ -n "$SYM_CKPT" ]; then
    run_evaluation "$STAGE_SYM" "$AT_SYM_CFG" "$SYM_CKPT" "$SYM_EVAL_DIR" || true
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

append_summary "$STAGE_SYM" "$SYM_TRAIN_DIR" "$SYM_EVAL_DIR" "$SYM_CKPT" "$SYM_TRAIN_STATUS" "$SYM_EVAL_STATUS"

# ---------------------------------------------------------------------------
# Stage 2: AT temporal-only (T=20)
# ---------------------------------------------------------------------------
STAGE_TEMP="at_temporal_only"
TEMP_TRAIN_DIR="$ROOT_DIR/$STAGE_TEMP/train"
TEMP_EVAL_DIR="$ROOT_DIR/$STAGE_TEMP/eval"
TEMP_CKPT_DIR="$CHECKPOINT_ROOT/$STAGE_TEMP"

run_training "$STAGE_TEMP" "$AT_TEMP_CFG" "$TEMP_TRAIN_DIR" \
    "training.checkpoint_dir=$TEMP_CKPT_DIR distillation.teacher_checkpoint=$TEACHER_CKPT logging.run_name=$STAGE_TEMP" || true

TEMP_CKPT="$(pick_best_ckpt "$TEMP_CKPT_DIR")"
if [ -n "$TEMP_CKPT" ]; then
    run_evaluation "$STAGE_TEMP" "$AT_TEMP_CFG" "$TEMP_CKPT" "$TEMP_EVAL_DIR" || true
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

append_summary "$STAGE_TEMP" "$TEMP_TRAIN_DIR" "$TEMP_EVAL_DIR" "$TEMP_CKPT" "$TEMP_TRAIN_STATUS" "$TEMP_EVAL_STATUS"

# ---------------------------------------------------------------------------
# Split t-SNE plots
# ---------------------------------------------------------------------------
run_tsne "tsne_teacher_baseline_distill_at_symmetric" "$SYM_CKPT" || true
run_tsne "tsne_teacher_baseline_distill_at_temporal" "$TEMP_CKPT" || true

# ---------------------------------------------------------------------------
# Predictions (ONLY for the two AT runs requested)
# - Do NOT produce confusion matrices; save predictions CSV only.
# ---------------------------------------------------------------------------
run_predictions "at_symmetric_t20" "student" "$STUDENT_CONFIG" "$SYM_CKPT" || true
run_predictions "at_temporal_t20" "student" "$STUDENT_CONFIG" "$TEMP_CKPT" || true

echo ""
echo "Pipeline completed. Summary: $SUMMARY_FILE"
echo "Predictions directory: $PRED_DIR"
echo "t-SNE directory: $TSNE_DIR"

if [ "$overall_status" != "SUCCESS" ]; then
    exit 1
fi
