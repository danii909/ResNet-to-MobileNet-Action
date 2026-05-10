#!/bin/bash
# ============================================================================
# SLURM batch script - Run multiple trainings sequentially in ONE job.
#
# Use this when QoS allows only one submitted job per user.
#
# Usage:
#   CONFIGS="experiments/configs/teacher.yaml experiments/configs/baseline.yaml experiments/configs/distillation.yaml" \
#   sbatch cluster/train_sequential.sh
#
# Optional:
#   EXTRA_ARGS="--override training.batch_size=16"
# ============================================================================

#SBATCH --job-name=kd-train-seq
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-train-seq-%j.log

set -euo pipefail

CONFIGS="${CONFIGS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

infer_training_type() {
    local cfg_lower
    cfg_lower=$(basename "$1" | tr '[:upper:]' '[:lower:]')
    if [[ "$cfg_lower" == *"teacher"* ]]; then
        echo "teacher"
    elif [[ "$cfg_lower" == *"baseline"* ]]; then
        echo "baseline"
    elif [[ "$cfg_lower" == *"distill"* ]]; then
        echo "distillation"
    else
        echo "unknown"
    fi
}

GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
GPU_NAME="${GPU_NAME:-unknown}"

if [ -z "$CONFIGS" ]; then
    echo "CONFIGS is empty. Example:"
    echo "  CONFIGS=\"experiments/configs/teacher.yaml experiments/configs/baseline.yaml\" sbatch cluster/train_sequential.sh"
    exit 1
fi

echo "============================================"
echo "  KD Sequential Training - Cluster DMI"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Node:      $(hostname)"
echo "  GPU:       ${GPU_NAME}"
echo "  Date:      $(date)"
echo "  Configs:   ${CONFIGS}"
echo "  Extra:     ${EXTRA_ARGS}"
echo "============================================"

mkdir -p logs

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

cd "$HOME/dl26-projects"

# One parent directory for the sequential job; each run gets a subfolder.
PARENT_TAG="slurm-train-seq-${SLURM_JOB_ID:-nojob}"
PARENT_DIR="$HOME/dl26-projects/experiments/logs/${PARENT_TAG}"
mkdir -p "$PARENT_DIR"

echo "parent_log_dir=${PARENT_DIR}" > "$PARENT_DIR/job_meta.txt"
echo "job_tag=${PARENT_TAG}" >> "$PARENT_DIR/job_meta.txt"
echo "slurm_job_id=${SLURM_JOB_ID}" >> "$PARENT_DIR/job_meta.txt"
echo "partition=${SLURM_JOB_PARTITION:-}" >> "$PARENT_DIR/job_meta.txt"
echo "qos=${SLURM_JOB_QOS:-}" >> "$PARENT_DIR/job_meta.txt"
echo "hostname=$(hostname)" >> "$PARENT_DIR/job_meta.txt"
echo "gpu_name=${GPU_NAME}" >> "$PARENT_DIR/job_meta.txt"
echo "configs=${CONFIGS}" >> "$PARENT_DIR/job_meta.txt"
echo "extra_args=${EXTRA_ARGS}" >> "$PARENT_DIR/job_meta.txt"
echo "started_at=$(date --iso-8601=seconds)" >> "$PARENT_DIR/job_meta.txt"

idx=1
total=$(echo "$CONFIGS" | wc -w)

for cfg in $CONFIGS; do
    if [ ! -f "$cfg" ]; then
        echo "Config not found: $cfg"
        exit 1
    fi

    run_tag="run-${idx}"
    run_dir="$PARENT_DIR/$run_tag"
    mkdir -p "$run_dir"
    training_type="$(infer_training_type "$cfg")"
    export TRAIN_LOG_DIR="$run_dir"
    export TRAIN_CONFIG_PATH="$cfg"
    export TRAINING_TYPE="$training_type"

    echo ""
    echo "[${idx}/${total}] Starting: $cfg"
    echo "run_config=${cfg}" > "$run_dir/job_meta.txt"
    echo "run_index=${idx}" >> "$run_dir/job_meta.txt"
    echo "training_type=${training_type}" >> "$run_dir/job_meta.txt"
    echo "hostname=$(hostname)" >> "$run_dir/job_meta.txt"
    echo "gpu_name=${GPU_NAME}" >> "$run_dir/job_meta.txt"
    echo "started_at=$(date --iso-8601=seconds)" >> "$run_dir/job_meta.txt"

    set +e
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env HF_DATASETS_OFFLINE=1 \
        /shared/sifs/latest.sif \
        python -u -m src.training.train --config "$cfg" ${EXTRA_ARGS}
    rc=$?
    set -e

    echo "finished_at=$(date --iso-8601=seconds)" >> "$run_dir/job_meta.txt"
    echo "exit_code=${rc}" >> "$run_dir/job_meta.txt"

    if [ "$rc" -ne 0 ]; then
        touch "$run_dir/status_FAILED"
        echo "[${idx}/${total}] FAILED: $cfg (exit code ${rc})"
        echo "finished_at=$(date --iso-8601=seconds)" >> "$PARENT_DIR/job_meta.txt"
        echo "overall_status=FAILED" >> "$PARENT_DIR/job_meta.txt"
        touch "$PARENT_DIR/status_FAILED"
        python3 -u -m src.utils.job_summary --job-log-dir "$PARENT_DIR" --quiet || true
        exit "$rc"
    fi

    touch "$run_dir/status_SUCCESS"
    python3 -u -m src.utils.job_summary --job-log-dir "$PARENT_DIR" --quiet || true
    echo "[${idx}/${total}] SUCCESS: $cfg"
    idx=$((idx + 1))
done

echo "finished_at=$(date --iso-8601=seconds)" >> "$PARENT_DIR/job_meta.txt"
echo "overall_status=SUCCESS" >> "$PARENT_DIR/job_meta.txt"
touch "$PARENT_DIR/status_SUCCESS"

SLURM_STDOUT_FILE="${SLURM_SUBMIT_DIR}/logs/slurm-train-seq-${SLURM_JOB_ID}.log"
if [ -f "$SLURM_STDOUT_FILE" ]; then
    cp "$SLURM_STDOUT_FILE" "$PARENT_DIR/slurm-stdout.log"
fi

python3 -u -m src.utils.job_summary --job-log-dir "$PARENT_DIR" --quiet || true

echo ""
echo "============================================"
echo "  Sequential training completed!"
echo "  $(date)"
echo "============================================"
