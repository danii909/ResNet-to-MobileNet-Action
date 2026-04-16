#!/bin/bash
# ============================================================================
# Single-job recovery run: KD T10/A0.7 with 24f and conservative augmentations.
#
# Usage:
#   sbatch cluster/submit_single_job_recovery24f.sh
# Optional:
#   TEACHER_CKPT=/path/to/teacher_finetune_best.pth sbatch cluster/submit_single_job_recovery24f.sh
# ============================================================================

#SBATCH --job-name=kd-recovery-24f
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-recovery-24f-%j.log

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
TEACHER_CKPT="${TEACHER_CKPT:-$PROJECT_DIR/experiments/checkpoints/teacher_finetune_best.pth}"
RUN_NAME="kd_t10_a07_24f_lightaug"
CFG="experiments/configs/distillation_t10_a07_24f_lightaug.yaml"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory not found: $PROJECT_DIR"
    exit 1
fi
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "Teacher checkpoint not found: $TEACHER_CKPT"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs

ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-recovery-24f-${SLURM_JOB_ID:-local}"
TRAIN_DIR="$ROOT_DIR/$RUN_NAME/train"
EVAL_DIR="$ROOT_DIR/$RUN_NAME/eval"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-recovery-24f-${SLURM_JOB_ID:-local}/$RUN_NAME"
CHECKPOINT_PATH="$CHECKPOINT_ROOT/distillation_best.pth"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"

mkdir -p "$TRAIN_DIR" "$EVAL_DIR" "$CHECKPOINT_ROOT"

{
    echo "Pipeline Summary"
    echo "============================================================"
    echo "job_tag: slurm-recovery-24f-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo "teacher_checkpoint: $TEACHER_CKPT"
    echo ""
    echo "[$RUN_NAME]"
    echo "train_dir: $TRAIN_DIR"
    echo "eval_dir: $EVAL_DIR"
} > "$SUMMARY_FILE"

echo ">>> TRAINING: $RUN_NAME"
set +e
TRAIN_LOG_DIR="$TRAIN_DIR" \
TRAIN_CONFIG_PATH="$CFG" \
TRAINING_TYPE="$RUN_NAME" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    /shared/sifs/latest.sif \
    python -u -m src.training.train --config "$CFG" --override \
      "distillation.teacher_checkpoint=$TEACHER_CKPT" \
      "training.checkpoint_dir=$CHECKPOINT_ROOT" \
      "logging.run_name=$RUN_NAME"
TRAIN_RC=$?
set -e

if [ "$TRAIN_RC" -eq 0 ]; then
    touch "$TRAIN_DIR/status_SUCCESS"
    echo "train_status: SUCCESS" >> "$SUMMARY_FILE"
else
    touch "$TRAIN_DIR/status_FAILED"
    echo "train_status: FAILED" >> "$SUMMARY_FILE"
fi
printf 'exit_code=%s\n' "$TRAIN_RC" > "$TRAIN_DIR/status.txt"

if [ "$TRAIN_RC" -ne 0 ] || [ ! -f "$CHECKPOINT_PATH" ]; then
    touch "$EVAL_DIR/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$EVAL_DIR/status.txt"
    echo "eval_status: FAILED" >> "$SUMMARY_FILE"
    echo "checkpoint_missing_or_train_failed" >> "$SUMMARY_FILE"
    exit 1
fi

echo ">>> EVALUATING: $RUN_NAME"
set +e
EVAL_LOG_DIR="$EVAL_DIR" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.evaluate --config "$CFG" --override "evaluation.checkpoint=$CHECKPOINT_PATH"
EVAL_RC=$?
set -e

if [ "$EVAL_RC" -eq 0 ]; then
    touch "$EVAL_DIR/status_SUCCESS"
    echo "eval_status: SUCCESS" >> "$SUMMARY_FILE"
else
    touch "$EVAL_DIR/status_FAILED"
    echo "eval_status: FAILED" >> "$SUMMARY_FILE"
fi
printf 'exit_code=%s\n' "$EVAL_RC" > "$EVAL_DIR/status.txt"

echo ""
echo "Run completed."
echo "Summary: $SUMMARY_FILE"

if [ "$EVAL_RC" -ne 0 ]; then
    exit "$EVAL_RC"
fi
