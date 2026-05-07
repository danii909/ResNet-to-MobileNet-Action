#!/bin/bash
#SBATCH --job-name=kd-at-4409
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-at-4409-%j.log

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
cd "$PROJECT_DIR"

mkdir -p logs

apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    /shared/sifs/latest.sif \
    python -u -m src.training.train --config experiments/configs/distillation_at_4409.yaml
