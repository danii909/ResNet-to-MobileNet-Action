#!/bin/bash
# ============================================================================
# Single job:
# 1) Train KD T=8 on 4495 split with label_smoothing=0.05
# 2) Generate predictions CSVs on test set for group-aware models
# ============================================================================

#SBATCH --job-name=kd-t8-ls005-preds
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm-kd-t8-ls005-preds-%j.log

set -euo pipefail

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

ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-train-eval-${SLURM_JOB_ID:-local}"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-${SLURM_JOB_ID:-local}"
PRED_DIR="$ROOT_DIR/predictions"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT" "$PRED_DIR"

BASE_CONFIG="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
TEACHER_4495="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth"

run_training() {
    local phase="$1"
    local config="$2"
    local train_dir="$3"
    local overrides="$4"

    mkdir -p "$train_dir"

    local -a ov=()
    if [ -n "$overrides" ]; then
        read -r -a ov <<< "$overrides"
    fi

    echo ""
    echo ">>> TRAINING: $phase"
    echo "    config: $config"

    set +e
    TRAIN_LOG_DIR="$train_dir" \
    TRAIN_CONFIG_PATH="$config" \
    TRAINING_TYPE="$phase" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.training.train --config "$config" --override "${ov[@]}"
    local rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        echo "Training failed for $phase (exit_code=$rc)."
        return "$rc"
    fi
}

run_predictions() {
    local label="$1"
    local model_type="$2"
    local config="$3"
    local checkpoint="$4"

    if [ ! -f "$checkpoint" ]; then
        echo "WARNING: missing checkpoint for $label: $checkpoint"
        return 1
    fi

    echo ""
    echo ">>> PREDICTIONS: $label"

    set +e
    PREDICTIONS_LOG_DIR="$PRED_DIR" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.confusion \
            --config "$config" \
            --override \
                "model.type=$model_type" \
                "evaluation.checkpoint=$checkpoint" \
                "evaluation.label=$label" \
                "evaluation.save_predictions_only=true" \
                "evaluation.predictions_dir=$PRED_DIR"
    local rc=$?
    set -e

    return "$rc"
}

# ---------------------------------------------------------------------------
# 1) KD T=8 with label_smoothing=0.05 (4495 split settings)
# ---------------------------------------------------------------------------
KD_PHASE="kd_t8_a07_24f_lightaug_ls005"
KD_TRAIN_DIR="$ROOT_DIR/$KD_PHASE/train"
KD_CKPT_DIR="$CHECKPOINT_ROOT/$KD_PHASE"
KD_CKPT="$KD_CKPT_DIR/distillation_best.pth"

KD_OVERRIDES="training.label_smoothing=0.05 distillation.teacher_checkpoint=$TEACHER_4495 training.checkpoint_dir=$KD_CKPT_DIR logging.run_name=$KD_PHASE"

run_training "$KD_PHASE" "$BASE_CONFIG" "$KD_TRAIN_DIR" "$KD_OVERRIDES" || true

# ---------------------------------------------------------------------------
# 2) Predictions on test set for group-aware models
# ---------------------------------------------------------------------------
STUDENT_CONFIG="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
TEACHER_CONFIG="experiments/configs/teacher_24f_evalsplit.yaml"

run_predictions "teacher_24f_lightaug" "teacher" "$TEACHER_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth"
run_predictions "baseline_ls005_24f_lightaug" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth"
run_predictions "kd_t8_a07_24f_lightaug" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/kd_t8_a07_24f_lightaug/distillation_best.pth"

run_predictions "kd_t1_a07_24f_lightaug" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4565/kd_t1_a07_24f_lightaug/distillation_best.pth"
run_predictions "kd_t5_a07_24f_lightaug" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4565/kd_t5_a07_24f_lightaug/distillation_best.pth"
run_predictions "kd_t10_a07_24f_lightaug" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4565/kd_t10_a07_24f_lightaug/distillation_best.pth"
run_predictions "kd_t20_a07_24f_lightaug" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4565/kd_t20_a07_24f_lightaug/distillation_best.pth"

run_predictions "at_symmetric" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4511/at_symmetric/distillation_at_best.pth"
run_predictions "at_temporal_only" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4511/at_temporal_only/distillation_at_best.pth"

run_predictions "born_again_gen1" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-born-again-4529/born_again_gen1/distillation_at_best.pth"
run_predictions "born_again_gen2" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-born-again-4529/born_again_gen2/distillation_at_best.pth"
run_predictions "born_again_gen3" "student" "$STUDENT_CONFIG" \
    "$PROJECT_DIR/experiments/checkpoints/slurm-born-again-4529/born_again_gen3/distillation_at_best.pth"

if [ -f "$KD_CKPT" ]; then
    run_predictions "$KD_PHASE" "student" "$STUDENT_CONFIG" "$KD_CKPT"
else
    echo "WARNING: KD checkpoint not found for $KD_PHASE ($KD_CKPT)."
fi

echo ""
echo "Done. Predictions saved in: $PRED_DIR"
