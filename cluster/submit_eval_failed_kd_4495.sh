#!/bin/bash
# ============================================================================
# Job per recuperare l'evaluation fallita del modello Distillation (Job 4495)
# ============================================================================

#SBATCH --job-name=eval-kd-4495
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm-eval-4495-retry-%j.log

set -euo pipefail

# FIX: Usa la home per i file temporanei
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
cd "$PROJECT_DIR"

export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

# Parametri del modello fallito
CONFIG="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
CHECKPOINT="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/kd_t8_a07_24f_lightaug/distillation_best.pth"
EVAL_DIR="$PROJECT_DIR/experiments/logs/slurm-train-eval-4495/kd_t8_a07_24f_lightaug/eval"

echo ">>> RETRYING EVALUATION: kd_t8_a07_24f_lightaug"
echo "    checkpoint: $CHECKPOINT"

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint non trovato in $CHECKPOINT"
    exit 1
fi

mkdir -p "$EVAL_DIR"

set +e
EVAL_LOG_DIR="$EVAL_DIR" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env TMPDIR="$TMPDIR" \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.evaluate --config "$CONFIG" --override "evaluation.checkpoint=$CHECKPOINT"
local_rc=$?
set -e

if [ "$local_rc" -eq 0 ]; then
    echo "Evaluation completata con successo!"
    touch "$EVAL_DIR/status_SUCCESS"
else
    echo "Evaluation fallita nuovamente con exit code $local_rc"
    touch "$EVAL_DIR/status_FAILED"
fi

echo "Done."
