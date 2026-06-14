#!/bin/bash
# ============================================================================
# SLURM batch script — Single Job Train & Eval sul cluster DMI
#
# Uso:
#   CONFIG=experiments/configs/valid/crossframe_kd_t24_s16.yaml sbatch cluster/train_eval_single.sh
# ============================================================================

#SBATCH --job-name=kd-single-train-eval
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-train-eval-%j.log

set -euo pipefail

# Avoid "No space left on device" in /tmp by using a local tmp directory
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

if [ -z "${CONFIG:-}" ]; then
    echo "❌ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/valid/crossframe_kd_t24_s16.yaml sbatch cluster/train_eval_single.sh"
    exit 1
fi

EXTRA_ARGS="${EXTRA_ARGS:-}"

JOB_TAG="slurm-train-eval-${SLURM_JOB_ID:-local}"
JOB_LOG_DIR="$PROJECT_DIR/experiments/logs/${JOB_TAG}"
JOB_CKPT_DIR="$PROJECT_DIR/experiments/checkpoints/${JOB_TAG}"

mkdir -p "$JOB_LOG_DIR" "$JOB_CKPT_DIR"

TRAIN_LOG_DIR="$JOB_LOG_DIR/train"
EVAL_LOG_DIR="$JOB_LOG_DIR/eval"
mkdir -p "$TRAIN_LOG_DIR" "$EVAL_LOG_DIR"

# 1) Esegui Training
echo "============================================"
# Training info
echo "  [1/2] TRAINING"
echo "  Job ID:        ${SLURM_JOB_ID:-local}"
echo "  Node:          $(hostname)"
echo "  Config:        ${CONFIG}"
echo "  Log Dir:       ${TRAIN_LOG_DIR}"
echo "  Checkpoint Dir:${JOB_CKPT_DIR}"
echo "============================================"

set +e
TRAIN_LOG_DIR="$TRAIN_LOG_DIR" \
TRAIN_CONFIG_PATH="$CONFIG" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env PYTHONUNBUFFERED=1 \
    --env TMPDIR="$TMPDIR" \
    /shared/sifs/latest.sif \
    python -u -m src.training.train --config "$CONFIG" --override training.checkpoint_dir="$JOB_CKPT_DIR" ${EXTRA_ARGS}
TRAIN_EXIT_CODE=$?
set -e

if [ "$TRAIN_EXIT_CODE" -eq 0 ]; then
    touch "$TRAIN_LOG_DIR/status_SUCCESS"
else
    touch "$TRAIN_LOG_DIR/status_FAILED"
    echo "❌ Training fallito con exit code ${TRAIN_EXIT_CODE}."
    exit "$TRAIN_EXIT_CODE"
fi

# Trova il checkpoint migliore appena salvato
CHECKPOINT=$(ls "${JOB_CKPT_DIR}"/*_best.pth 2>/dev/null | head -n 1 || true)

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Impossibile trovare il checkpoint migliore in ${JOB_CKPT_DIR}."
    exit 1
fi

# 2) Esegui Evaluation sul Test Set
# Determina se c'è un override per student_frames (Cross-Frame KD)
# Se sì, dobbiamo impostare dataset.num_frames al valore di student_frames per valutare correttamente lo student a 16f.
STUDENT_FRAMES=$(python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('distillation', {}).get('student_frames', ''))
" 2>/dev/null || true)

EVAL_OVERRIDE="evaluation.checkpoint=${CHECKPOINT}"
if [ -n "$STUDENT_FRAMES" ]; then
    echo "ℹ️ Rilevata configurazione Cross-Frame KD: imposto dataset.num_frames=${STUDENT_FRAMES} per l'evaluation dello Student."
    EVAL_OVERRIDE="${EVAL_OVERRIDE} dataset.num_frames=${STUDENT_FRAMES}"
fi

echo ""
echo "============================================"
echo "  [2/2] EVALUATION"
echo "  Config:        ${CONFIG}"
echo "  Checkpoint:    ${CHECKPOINT}"
echo "  Log Dir:       ${EVAL_LOG_DIR}"
echo "============================================"

set +e
EVAL_LOG_DIR="$EVAL_LOG_DIR" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    --env TMPDIR="$TMPDIR" \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.evaluate --config "$CONFIG" --override ${EVAL_OVERRIDE}
EVAL_EXIT_CODE=$?
set -e

SLURM_STDOUT_FILE="${SLURM_SUBMIT_DIR:-$PROJECT_DIR}/logs/slurm-train-eval-${SLURM_JOB_ID:-local}.log"
if [ -f "$SLURM_STDOUT_FILE" ]; then
    cp "$SLURM_STDOUT_FILE" "$JOB_LOG_DIR/slurm-stdout.log" || true
fi

if [ "$EVAL_EXIT_CODE" -eq 0 ]; then
    touch "$EVAL_LOG_DIR/status_SUCCESS"
    echo "✅ Pipeline completata con successo!"
else
    touch "$EVAL_LOG_DIR/status_FAILED"
    echo "❌ Evaluation fallita con exit code ${EVAL_EXIT_CODE}."
    exit "$EVAL_EXIT_CODE"
fi
