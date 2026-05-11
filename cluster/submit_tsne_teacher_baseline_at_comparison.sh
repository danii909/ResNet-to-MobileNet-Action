#!/bin/bash
# ============================================================================
# SLURM script to generate t-SNE visualization comparing:
# - Teacher and Baseline from run 4495
# - AT Symmetric and AT Temporal-only from run 4511
# Visualization only: no training, no evaluation.
# ============================================================================

#SBATCH --job-name=tsne-teacher-baseline-at
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-tsne-teacher-baseline-at-%j.log

set -euo pipefail

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
cd "$PROJECT_DIR"
mkdir -p logs

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

# Checkpoints from run 4495 (baseline and teacher)
TEACHER_CKPT="experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth"
BASELINE_CKPT="experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth"

# AT checkpoints from run 4511
AT_SYM_CKPT="experiments/checkpoints/slurm-train-eval-4511/at_symmetric/distillation_at_best.pth"
AT_TEMP_CKPT="experiments/checkpoints/slurm-train-eval-4511/at_temporal_only/distillation_at_best.pth"

CONFIG="experiments/configs/at_symmetric_p50.yaml"
OUTPUT_DIR="experiments/logs/tsne_teacher_baseline_at_comparison"
mkdir -p "$OUTPUT_DIR"

echo "Starting t-SNE visualization: Teacher vs Baseline vs AT models"
echo "========================================================================"
echo "Config: $CONFIG"
echo "Teacher (4495): $TEACHER_CKPT"
echo "Baseline (4495): $BASELINE_CKPT"
echo "AT Symmetric (4511): $AT_SYM_CKPT"
echo "AT Temporal-only (4511): $AT_TEMP_CKPT"
echo "Output dir: $OUTPUT_DIR"
echo "========================================================================"

# Verify all checkpoints exist
for ckpt in "$TEACHER_CKPT" "$BASELINE_CKPT" "$AT_SYM_CKPT" "$AT_TEMP_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Missing checkpoint: $ckpt"
        exit 1
    fi
done

set +e
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env TMPDIR="$TMPDIR" \
    --env PYTHONUNBUFFERED=1 \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.tsne_visualizer \
        --config "$CONFIG" \
        --teacher-ckpt "$TEACHER_CKPT" \
        --baseline-ckpt "$BASELINE_CKPT" \
        --student-ckpt "$AT_SYM_CKPT" \
        --student2-ckpt "$AT_TEMP_CKPT" \
        --num-classes 10 \
        --output-dir "$OUTPUT_DIR" \
        --output-filename "tsne_teacher_baseline_at_comparison"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    echo "========================================================================"
    echo "t-SNE visualization completed!"
    echo "Plots saved in: $OUTPUT_DIR"
    echo "========================================================================"
else
    echo "ERROR: t-SNE visualization failed with exit code $rc"
    exit "$rc"
fi
