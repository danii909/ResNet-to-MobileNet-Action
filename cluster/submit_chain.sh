#!/bin/bash
# ============================================================================
# Submit a sequence of training jobs with SLURM dependencies.
#
# Runs one job at a time: each job starts only if the previous finished OK.
# This is the compliant way to queue multiple trainings with a single active slot.
#
# Usage:
#   bash cluster/submit_chain.sh \
#     experiments/configs/teacher.yaml \
#     experiments/configs/baseline.yaml \
#     experiments/configs/distillation.yaml
#
# Optional:
#   EXTRA_ARGS="--override training.batch_size=16" bash cluster/submit_chain.sh ...
# ============================================================================

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash cluster/submit_chain.sh <config1.yaml> [config2.yaml ...]"
  echo "Example:"
  echo "  bash cluster/submit_chain.sh experiments/configs/teacher.yaml experiments/configs/baseline.yaml"
  exit 1
fi

for cfg in "$@"; do
  if [ ! -f "$cfg" ]; then
    echo "Config not found: $cfg"
    exit 1
  fi
done

mkdir -p logs

prev_job_id=""
idx=1

echo "Submitting chain of $# training job(s)..."
for cfg in "$@"; do
  if [ -z "$prev_job_id" ]; then
    job_id=$(CONFIG="$cfg" EXTRA_ARGS="${EXTRA_ARGS:-}" sbatch --parsable cluster/train.sh)
    echo "[$idx/$#] Submitted: $cfg -> job $job_id"
  else
    job_id=$(CONFIG="$cfg" EXTRA_ARGS="${EXTRA_ARGS:-}" sbatch --parsable --dependency=afterok:$prev_job_id cluster/train.sh)
    echo "[$idx/$#] Submitted: $cfg -> job $job_id (afterok:$prev_job_id)"
  fi

  prev_job_id="$job_id"
  idx=$((idx + 1))
done

echo ""
echo "Done. Last job in chain: $prev_job_id"
echo "Check queue with: squeue --me"
