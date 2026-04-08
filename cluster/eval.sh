#!/bin/bash
# ============================================================================
# SLURM batch script â€” Evaluation sul cluster DMI
#
# Uso:
#   CONFIG=experiments/configs/baseline.yaml CHECKPOINT=experiments/checkpoints/baseline_best.pth sbatch cluster/eval.sh
#   CONFIG=experiments/configs/teacher.yaml CHECKPOINT=experiments/checkpoints/teacher_finetune_best.pth sbatch cluster/eval.sh
#   CONFIG=experiments/configs/distillation.yaml CHECKPOINT=experiments/checkpoints/distillation_best.pth sbatch cluster/eval.sh
# ============================================================================

# â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
# â”‚  CONFIGURA QUI â€” modifica account/partition/qos/email  â”‚
# â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
#SBATCH --job-name=kd-eval
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-eval-%j.log

# â”€â”€ Variabili progetto â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
CHECKPOINT="${CHECKPOINT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [ -z "$CONFIG" ]; then
    echo "âŒ CONFIG non impostato. Uso:"
    echo "  CONFIG=experiments/configs/baseline.yaml CHECKPOINT=experiments/checkpoints/baseline_best.pth sbatch cluster/eval.sh"
    echo ""
    echo "Config disponibili:"
    ls -1 experiments/configs/*.yaml 2>/dev/null | sed 's/^/  /'
    exit 1
fi

if [ -z "$CHECKPOINT" ]; then
    echo "âŒ CHECKPOINT non impostato. Uso:"
    echo "  CHECKPOINT=experiments/checkpoints/baseline_best.pth CONFIG=... sbatch cluster/eval.sh"
    exit 1
fi

# â”€â”€ Setup ambiente â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
set -e

echo "============================================"
echo "  KD Evaluation â€” Cluster DMI"
echo "  Job ID:      ${SLURM_JOB_ID}"
echo "  Node:        $(hostname)"
echo "  Date:        $(date)"
echo "  Config:      ${CONFIG}"
echo "  Checkpoint:  ${CHECKPOINT}"
echo "============================================"

mkdir -p logs

export WANDB_MODE=offline
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

cd "$HOME/dl26-projects"

EVAL_ARGS="--config ${CONFIG} --override evaluation.checkpoint=${CHECKPOINT}"
if [ -n "$EXTRA_ARGS" ]; then
    EVAL_ARGS="${EVAL_ARGS} ${EXTRA_ARGS}"
fi

echo ""
echo "Avvio evaluation dentro Apptainer..."
echo "  python -m src.evaluation.evaluate ${EVAL_ARGS}"
echo ""

# â”€â”€ Esecuzione dentro container Apptainer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    /shared/sifs/latest.sif \
    python -m src.evaluation.evaluate ${EVAL_ARGS}

echo ""
echo "============================================"
echo "  Evaluation completata!"
echo "  $(date)"
echo "============================================"
