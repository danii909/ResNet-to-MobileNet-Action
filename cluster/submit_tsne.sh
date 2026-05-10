#!/bin/bash
# ============================================================================
# SLURM script to generate t-SNE latent space visualizations
# ============================================================================

#SBATCH --job-name=tsne-visualize
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-tsne-%j.log

set -euo pipefail

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

# Usage: sbatch cluster/submit_tsne.sh <config> <teacher_ckpt> <student_ckpt> [baseline_ckpt] [output_filename] [num_classes]
CONFIG="${1:-experiments/configs/distillation_t8_a07_24f_lightaug.yaml}"
TEACHER_CKPT="${2:-experiments/checkpoints/teacher_finetune_best.pth}"
STUDENT_CKPT="${3:-experiments/checkpoints/distillation_best.pth}"
BASELINE_CKPT="${4:-}"
OUTPUT_FILENAME="${5:-tsne_comparison_teacher_vs_student}"
NUM_CLASSES="${6:-10}"

echo "Starting t-SNE Visualization"
echo "Config: $CONFIG"
echo "Teacher Checkpoint: $TEACHER_CKPT"
echo "Student Checkpoint: $STUDENT_CKPT"
if [ -n "$BASELINE_CKPT" ]; then
    echo "Baseline Checkpoint: $BASELINE_CKPT"
fi
echo "Output Filename: $OUTPUT_FILENAME"
echo "Number of Classes: $NUM_CLASSES"
echo "========================================="

# Costruiamo gli argomenti base
TSNE_ARGS=(
    "--config" "$CONFIG"
    "--teacher-ckpt" "$TEACHER_CKPT"
    "--student-ckpt" "$STUDENT_CKPT"
    "--num-classes" "$NUM_CLASSES"
    "--output-dir" "experiments/logs/tsne_plots"
    "--output-filename" "$OUTPUT_FILENAME"
)

# Aggiungiamo il baseline se presente
if [ -n "$BASELINE_CKPT" ]; then
    TSNE_ARGS+=("--baseline-ckpt" "$BASELINE_CKPT")
fi

set +e
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.tsne_visualizer "${TSNE_ARGS[@]}"
rc=$?
set -e

if [ "$rc" -eq 0 ]; then
    echo "t-SNE Visualization completed successfully."
else
    echo "t-SNE Visualization failed with exit code $rc."
    exit $rc
fi
