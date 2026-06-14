#!/bin/bash
# ============================================================================
# SLURM batch script — Compare 24f and 16f inference latency and throughput
#
# Uso:
#   sbatch cluster/benchmark_comparison.sh
# ============================================================================

#SBATCH --job-name=kd-benchmark
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-benchmark-%j.log

set -euo pipefail

# Setup environment
export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs

echo "============================================"
echo "  KD Inference Benchmark Comparison"
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Node:        $(hostname)"
echo "  Date:        $(date)"
echo "============================================"
echo ""

# Run benchmark inside Apptainer
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.benchmark_comparison

echo ""
echo "============================================"
echo "  Benchmark completed!"
echo "  $(date)"
echo "============================================"
