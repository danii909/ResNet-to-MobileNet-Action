#!/bin/bash
# ============================================================================
# SLURM script to generate t-SNE for job 4495
# ============================================================================

#SBATCH --job-name=tsne-4495
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-tsne-4495-%j.log

set -euo pipefail

# FIX: Spazio temporaneo nella Home
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
cd "$PROJECT_DIR"

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

# Path specifici per il Job 4495
CONFIG="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
TEACHER_CKPT="experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth"
BASELINE_CKPT="experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth"
STUDENT_CKPT="experiments/checkpoints/slurm-train-eval-4495/kd_t8_a07_24f_lightaug/distillation_best.pth"

OUTPUT_DIR="experiments/logs/slurm-train-eval-4495/tsne_plots"
mkdir -p "$OUTPUT_DIR"

echo "Starting t-SNE for Job 4495"
echo "Teacher: $TEACHER_CKPT"
echo "Baseline: $BASELINE_CKPT"
echo "Student: $STUDENT_CKPT"

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
        --student-ckpt "$STUDENT_CKPT" \
        --num-classes 10 \
        --output-dir "$OUTPUT_DIR" \
        --output-filename "tsne_comparison_4495"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    echo "t-SNE completed! Plot saved in $OUTPUT_DIR"
else
    echo "t-SNE failed with exit code $rc"
    exit $rc
fi
