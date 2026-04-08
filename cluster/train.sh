#!/bin/bash
# ============================================================================
# SLURM batch script â€” Training sul cluster DMI
#
# Uso:
#   CONFIG=experiments/configs/teacher.yaml sbatch cluster/train.sh
#   CONFIG=experiments/configs/distillation.yaml EXTRA_ARGS="--override training.lr=0.005" sbatch cluster/train.sh
#
# Per il primo avvio eseguire prima: bash cluster/setup.sh
# (dentro una sessione interattiva Apptainer)
# ============================================================================

# â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
# â”‚  CONFIGURA QUI â€” modifica account/partition/qos/email  â”‚
# â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
#SBATCH --job-name=kd-train
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-train-%j.log

# â”€â”€ Variabili progetto â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$CONFIG" ]; then
    echo "âŒ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/teacher.yaml sbatch cluster/train.sh"
    echo ""
    echo "Config disponibili:"
    ls -1 experiments/configs/*.yaml 2>/dev/null | sed 's/^/  /'
    exit 1
fi

# â”€â”€ Setup ambiente â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
set -e

echo "============================================"
echo "  KD Training â€” Cluster DMI"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Config:    ${CONFIG}"
echo "  Extra:     ${EXTRA_ARGS}"
echo "============================================"

# Crea directory logs se non esiste
mkdir -p logs

# wandb in modalitÃ  offline (non c'Ã¨ internet sul cluster per studenti)
export WANDB_MODE=offline
# HF datasets offline (usa la cache scaricata da setup.sh)
export HF_DATASETS_OFFLINE=1
export HF_TOKEN="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"
# Memory management
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

# Percorso progetto
cd "$HOME/dl26-projects"

# Directory log strutturata per job (richiesta: experiments/logs/slurm-train-<jobid>)
JOB_TAG="slurm-train-${SLURM_JOB_ID:-nojob}"
JOB_LOG_DIR="$HOME/dl26-projects/experiments/logs/${JOB_TAG}"
mkdir -p "$JOB_LOG_DIR"
export TRAIN_LOG_DIR="$JOB_LOG_DIR"

# Salva metadati minimi del job in anticipo
{
    echo "job_tag=${JOB_TAG}"
    echo "slurm_job_id=${SLURM_JOB_ID}"
    echo "hostname=$(hostname)"
    echo "started_at=$(date --iso-8601=seconds)"
    echo "config=${CONFIG}"
    echo "extra_args=${EXTRA_ARGS}"
} > "$JOB_LOG_DIR/job_meta.txt"

# Verifica pesi pretrained
WEIGHTS_FILE="$HOME/dl26-projects/experiments/checkpoints/SLOW_8x8_R50.pyth"
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "ATTENZIONE: Pesi slow_r50 non trovati in $WEIGHTS_FILE"
    echo "   Caricali con: .\\sync_cluster.ps1 -Action upload"
else
    echo "Pesi slow_r50 presenti."
fi

echo ""
echo "Avvio training dentro Apptainer..."
echo "  python -m src.training.train --config ${CONFIG} ${EXTRA_ARGS}"
echo ""

# â”€â”€ Esecuzione dentro container Apptainer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
set +e
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    --env HF_DATASETS_OFFLINE=1 \
    --env HF_TOKEN="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" \
    /shared/sifs/latest.sif \
    python -u -m src.training.train --config "${CONFIG}" ${EXTRA_ARGS}
TRAIN_EXIT_CODE=$?
set -e

SLURM_STDOUT_FILE="${SLURM_SUBMIT_DIR}/logs/slurm-train-${SLURM_JOB_ID}.log"
if [ -f "$SLURM_STDOUT_FILE" ]; then
    cp "$SLURM_STDOUT_FILE" "$JOB_LOG_DIR/slurm-stdout.log"
fi

echo "finished_at=$(date --iso-8601=seconds)" >> "$JOB_LOG_DIR/job_meta.txt"
echo "exit_code=${TRAIN_EXIT_CODE}" >> "$JOB_LOG_DIR/job_meta.txt"

if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
    touch "$JOB_LOG_DIR/status_SUCCESS"
else
    touch "$JOB_LOG_DIR/status_FAILED"
fi

echo ""
echo "============================================"
echo "  Training completato!"
echo "  $(date)"
echo "============================================"

if [ "$TRAIN_EXIT_CODE" -ne 0 ]; then
    echo "Training fallito con exit code ${TRAIN_EXIT_CODE}."
    exit "$TRAIN_EXIT_CODE"
fi
