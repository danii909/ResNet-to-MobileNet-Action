#!/bin/bash
# ============================================================================
# Job per ri-eseguire solo le evaluation fallite dell'esperimento 4431
# ============================================================================

#SBATCH --job-name=eval-kd-4431
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm-eval-4431-%j.log

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs

export HF_DATASETS_OFFLINE=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

run_evaluation() {
    local phase="$1"
    local config="$2"
    local checkpoint="$3"
    local eval_dir="$4"

    mkdir -p "$eval_dir"

    echo ""
    echo ">>> EVALUATING: $phase"
    echo "    checkpoint: $checkpoint"

    set +e
    EVAL_LOG_DIR="$eval_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.evaluate --config "$config" --override "evaluation.checkpoint=$checkpoint"
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        touch "$eval_dir/status_SUCCESS"
    else
        touch "$eval_dir/status_FAILED"
    fi
    printf 'exit_code=%s\n' "$rc" > "$eval_dir/status.txt"

    return "$rc"
}

# --- Experiment 2: AT Temporal Only ---
exp2_cfg="experiments/configs/exp2_at_temporal_only.yaml"
exp2_eval_dir="$PROJECT_DIR/experiments/logs/slurm-train-eval-4431/exp2_at_temporal_only/eval"
exp2_ckpt="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4431/exp2_at_temporal_only/distillation_at_best.pth"

if [ -f "$exp2_ckpt" ]; then
    run_evaluation "exp2_at_temporal_only" "$exp2_cfg" "$exp2_ckpt" "$exp2_eval_dir"
else
    echo "Checkpoint non trovato: $exp2_ckpt"
fi

# --- Experiment 3: Late-Stage AT Semantico ---
exp3_cfg="experiments/configs/exp3_at_latestage_semantico.yaml"
exp3_eval_dir="$PROJECT_DIR/experiments/logs/slurm-train-eval-4431/exp3_at_latestage_semantico/eval"
exp3_ckpt="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4431/exp3_at_latestage_semantico/distillation_at_best.pth"

if [ -f "$exp3_ckpt" ]; then
    run_evaluation "exp3_at_latestage_semantico" "$exp3_cfg" "$exp3_ckpt" "$exp3_eval_dir"
else
    echo "Checkpoint non trovato: $exp3_ckpt"
fi

echo ""
echo "Evaluation job completed."
