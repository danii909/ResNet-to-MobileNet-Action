#!/bin/bash
# ============================================================================
# Evaluate a checkpoint on the test set and save outputs into exp 4577 eval
# folder, plus predictions CSV for confusion matrix.
#
# Optional overrides:
#   CONFIG=experiments/configs/distillation_t8_a07_24f_lightaug.yaml \
#   CHECKPOINT=/home/.../distillation_best.pth \
#   EVAL_DIR=/home/.../experiments/logs/slurm-train-eval-4577/.../eval \
#   LABEL=kd_t8_a07_24f_lightaug_ls005_4577 \
#   sbatch cluster/submit_eval_4577_with_preds.sh
# ============================================================================

#SBATCH --job-name=eval-4577-with-preds
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm-eval-4577-with-preds-%j.log

set -euo pipefail

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

CONFIG="${CONFIG:-experiments/configs/distillation_t8_a07_24f_lightaug.yaml}"
CHECKPOINT="${CHECKPOINT:-$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4577/kd_t8_a07_24f_lightaug_ls005/distillation_best.pth}"
EVAL_DIR="${EVAL_DIR:-$PROJECT_DIR/experiments/logs/slurm-train-eval-4577/kd_t8_a07_24f_lightaug_ls005/eval}"
LABEL="${LABEL:-kd_t8_a07_24f_lightaug_ls005_4577}"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    exit 1
fi

mkdir -p "$EVAL_DIR"

echo ""
echo ">>> EVALUATION"
echo "    config:     $CONFIG"
echo "    checkpoint: $CHECKPOINT"
echo "    eval dir:   $EVAL_DIR"

set +e
EVAL_LOG_DIR="$EVAL_DIR" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    --env TMPDIR="$TMPDIR" \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.evaluate \
        --config "$CONFIG" \
        --override "model.type=student" "evaluation.checkpoint=$CHECKPOINT"
EVAL_RC=$?
set -e

if [ "$EVAL_RC" -ne 0 ]; then
    echo "ERROR: evaluation failed (exit_code=$EVAL_RC)"
    exit "$EVAL_RC"
fi

echo ""
echo ">>> PREDICTIONS CSV"
echo "    label:      $LABEL"
echo "    output dir: $EVAL_DIR"

set +e
PREDICTIONS_LOG_DIR="$EVAL_DIR" \
CONFUSION_LOG_DIR="$EVAL_DIR" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    --env TMPDIR="$TMPDIR" \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.confusion \
        --config "$CONFIG" \
        --override \
            "model.type=student" \
            "evaluation.checkpoint=$CHECKPOINT" \
            "evaluation.label=$LABEL" \
            "evaluation.save_predictions_only=true" \
            "evaluation.predictions_dir=$EVAL_DIR"
PRED_RC=$?
set -e

if [ "$PRED_RC" -ne 0 ]; then
    echo "ERROR: predictions failed (exit_code=$PRED_RC)"
    exit "$PRED_RC"
fi

echo ""
echo "Done. Evaluation summary and predictions CSV saved in: $EVAL_DIR"
