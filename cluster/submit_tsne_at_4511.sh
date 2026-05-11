#!/bin/bash
# ============================================================================
# SLURM script to generate t-SNE for the AT checkpoints of run 4511
# Visualization only: no training, no evaluation.
# ============================================================================

#SBATCH --job-name=tsne-at-4511
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-tsne-at-4511-%j.log

set -euo pipefail

mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
cd "$PROJECT_DIR"
mkdir -p logs

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

CONFIG="experiments/configs/at_symmetric_p50.yaml"
TEACHER_CKPT="experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth"
BASELINE_CKPT="experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth"
AT_SYM_CKPT="experiments/checkpoints/slurm-train-eval-4511/at_symmetric/distillation_best.pth"
AT_TEMP_CKPT="experiments/checkpoints/slurm-train-eval-4511/at_temporal_only/distillation_best.pth"

OUTPUT_DIR="experiments/logs/slurm-train-eval-4511/tsne_plots"
mkdir -p "$OUTPUT_DIR"

echo "Starting t-SNE visualization for AT run 4511"
echo "Config: $CONFIG"
echo "Teacher: $TEACHER_CKPT"
echo "Baseline: $BASELINE_CKPT"
echo "AT symmetric: $AT_SYM_CKPT"
echo "AT temporal-only: $AT_TEMP_CKPT"
echo "Output dir: $OUTPUT_DIR"

for ckpt in "$TEACHER_CKPT" "$BASELINE_CKPT" "$AT_SYM_CKPT" "$AT_TEMP_CKPT"; do
    if [ ! -f "$ckpt" ]; then
        echo "Missing checkpoint: $ckpt"
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
        --output-filename "tsne_at_4511"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    echo "t-SNE completed! Plot saved in $OUTPUT_DIR"
else
    echo "t-SNE failed with exit code $rc"
    exit "$rc"
fi