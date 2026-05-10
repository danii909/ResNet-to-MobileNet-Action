#!/bin/bash
# ============================================================================
# Fast Test Job: Verify t-SNE visualization pipeline
# Runs a very short training (Teacher 4 epochs, Student 5 epochs)
# ============================================================================

#SBATCH --job-name=tsne-test-fast
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --output=logs/slurm-tsne-test-%j.log

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

ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-tsne-test-${SLURM_JOB_ID:-local}"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-tsne-test-${SLURM_JOB_ID:-local}"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT"

overall_status="SUCCESS"

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
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        /shared/sifs/latest.sif \
        python -u -m src.training.train --config "$config" --override "${ov[@]}"
    return $?
}

shared_overrides="dataset.use_eval_split=true dataset.eval_ratio=0.2 dataset.split_seed=42"

# 1) Fast Teacher Fine-tuning (4 epochs)
teacher_cfg="experiments/configs/teacher_24f_evalsplit.yaml"
teacher_phase="teacher_fast"
teacher_train_dir="$ROOT_DIR/$teacher_phase/train"
teacher_ckpt_dir="$CHECKPOINT_ROOT/$teacher_phase"
teacher_ckpt="$teacher_ckpt_dir/teacher_finetune_best.pth"

run_training "$teacher_phase" "$teacher_cfg" "$teacher_train_dir" \
    "$shared_overrides training.epochs=4 training.checkpoint_dir=$teacher_ckpt_dir logging.run_name=$teacher_phase" || overall_status="FAILED"

# 2) Fast Distillation (5 epochs, no warmup)
if [ "$overall_status" != "FAILED" ]; then
    kd_cfg="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
    kd_phase="kd_fast"
    kd_train_dir="$ROOT_DIR/$kd_phase/train"
    kd_ckpt_dir="$CHECKPOINT_ROOT/$kd_phase"
    kd_ckpt="$kd_ckpt_dir/distillation_best.pth"

    run_training "$kd_phase" "$kd_cfg" "$kd_train_dir" \
        "$shared_overrides training.epochs=5 training.kd_warmup_epochs=0 training.checkpoint_dir=$kd_ckpt_dir distillation.teacher_checkpoint=$teacher_ckpt logging.run_name=$kd_phase" || overall_status="FAILED"
fi

# 3) Latent Space t-SNE Visualization
echo ""
echo ">>> GENERATING t-SNE VISUALIZATION"
if [ "$overall_status" != "FAILED" ] && [ -f "$teacher_ckpt" ] && [ -f "$kd_ckpt" ]; then
    tsne_output_dir="$ROOT_DIR/tsne_plots"
    mkdir -p "$tsne_output_dir"
    
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.tsne_visualizer \
            --config "$kd_cfg" \
            --teacher-ckpt "$teacher_ckpt" \
            --student-ckpt "$kd_ckpt" \
            --num-classes 7 \
            --output-dir "$tsne_output_dir" \
            --output-filename "tsne_test_fast_${SLURM_JOB_ID:-local}"
    echo "t-SNE visualization generated in $tsne_output_dir"
else
    echo "Skipping t-SNE visualization because training failed or checkpoints are missing."
    exit 1
fi

echo ""
echo "Test pipeline completed. Summary: $SUMMARY_FILE"
echo "Overall status: $overall_status"
