#!/bin/bash
# ============================================================================
# Born Again Distillation: Student -> Student -> Student (3 generations)
# Strategy: Each generation student teaches the next generation
# Base teacher: distillation_best.pth from SLURM run 4201
# All generations use: distillation_t8_a07_24f_lightaug.yaml config
# ============================================================================

#SBATCH --job-name=born-again-distillation
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
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT"

overall_status="SUCCESS"

# Path to the initial teacher (from 4201)
# This is the distillation_best.pth from run 4201
# If not available locally, we'll use the remote path
INITIAL_TEACHER_4201="/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/slurm-train-eval-4201/distillation/distillation_best.pth"
INITIAL_TEACHER_LOCAL="$CHECKPOINT_ROOT/4201_distillation_teacher.pth"

# Copy or reference the 4201 teacher checkpoint
if [ ! -f "$INITIAL_TEACHER_LOCAL" ]; then
    echo "Attempting to locate 4201 distillation teacher..."
    if [ -f "$INITIAL_TEACHER_4201" ]; then
        cp "$INITIAL_TEACHER_4201" "$INITIAL_TEACHER_LOCAL"
        echo "Copied 4201 teacher from: $INITIAL_TEACHER_4201"
    elif [ -f "experiments/checkpoints/slurm-train-eval-4201/distillation/distillation_best.pth" ]; then
        cp "experiments/checkpoints/slurm-train-eval-4201/distillation/distillation_best.pth" "$INITIAL_TEACHER_LOCAL"
        echo "Copied 4201 teacher from local path"
    else
        echo "ERROR: Could not find 4201 distillation checkpoint!"
        echo "Expected at: $INITIAL_TEACHER_4201"
        overall_status="FAILED"
    fi
fi

run_born_again_training() {
    local generation="$1"          # 1, 2, 3
    local config="$2"              # distillation_t8_a07_24f_lightaug.yaml
    local teacher_ckpt="$3"        # Path to teacher checkpoint
    local train_dir="$4"           # Output training directory
    local student_ckpt_dir="$5"    # Output checkpoint directory
    local student_output_name="$6" # baseline_best.pth or distillation_best.pth

    mkdir -p "$train_dir" "$student_ckpt_dir"

    local phase_name="born_again_gen${generation}"
    local student_ckpt="$student_ckpt_dir/$student_output_name"

    echo ""
    echo ">>> BORN AGAIN GENERATION $generation"
    echo "    config: $config"
    echo "    teacher_checkpoint: $teacher_ckpt"
    echo "    student_output: $student_ckpt"

    if [ ! -f "$teacher_ckpt" ]; then
        echo "ERROR: Teacher checkpoint not found: $teacher_ckpt"
        touch "$train_dir/status_FAILED"
        printf 'exit_code=%s\n' 99 > "$train_dir/status.txt"
        overall_status="FAILED"
        return 1
    fi

    set +e
    TRAIN_LOG_DIR="$train_dir" \
    TRAIN_CONFIG_PATH="$config" \
    TRAINING_TYPE="distillation" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        /shared/sifs/latest.sif \
        python -u -m src.training.train \
            --config "$config" \
            --override \
                "dataset.use_eval_split=true" \
                "dataset.eval_ratio=0.2" \
                "dataset.split_seed=42" \
                "training.checkpoint_dir=$student_ckpt_dir" \
                "distillation.teacher_checkpoint=$teacher_ckpt" \
                "logging.run_name=$phase_name"
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        touch "$train_dir/status_SUCCESS"
    else
        touch "$train_dir/status_FAILED"
        overall_status="FAILED"
    fi
    printf 'exit_code=%s\n' "$rc" > "$train_dir/status.txt"

    return "$rc"
}

run_evaluation() {
    local generation="$1"
    local config="$2"
    local checkpoint="$3"
    local eval_dir="$4"

    mkdir -p "$eval_dir"

    echo ""
    echo ">>> EVALUATING GENERATION $generation"
    echo "    checkpoint: $checkpoint"

    set +e
    EVAL_LOG_DIR="$eval_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.evaluate \
            --config "$config" \
            --override "evaluation.checkpoint=$checkpoint"
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        touch "$eval_dir/status_SUCCESS"
    else
        touch "$eval_dir/status_FAILED"
        overall_status="FAILED"
    fi
    printf 'exit_code=%s\n' "$rc" > "$eval_dir/status.txt"

    return "$rc"
}

append_summary() {
    local generation="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local checkpoint="$4"

    local train_status="UNKNOWN"
    local eval_status="UNKNOWN"
    [ -f "$train_dir/status_SUCCESS" ] && train_status="SUCCESS"
    [ -f "$train_dir/status_FAILED" ] && train_status="FAILED"
    [ -f "$eval_dir/status_SUCCESS" ] && eval_status="SUCCESS"
    [ -f "$eval_dir/status_FAILED" ] && eval_status="FAILED"

    python3 - "$generation" "$train_dir" "$eval_dir" "$checkpoint" "$train_status" "$eval_status" >> "$SUMMARY_FILE" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_colon_kv(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or ":" not in line:
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    except Exception:
        return {}
    return out


def coalesce(d: dict[str, str], *keys: str, default: str = "n/a") -> str:
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return default


generation = sys.argv[1]
train_dir = Path(sys.argv[2])
eval_dir = Path(sys.argv[3])
checkpoint = sys.argv[4]
train_status = sys.argv[5]
eval_status = sys.argv[6]

cfg = read_json(train_dir / "config_summary.json")
train = read_colon_kv(train_dir / "training_summary.txt")
ev = read_json(eval_dir / "evaluation_summary.json")

model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
data_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
tr_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
kd_cfg = cfg.get("distillation", {}) if isinstance(cfg.get("distillation"), dict) else {}

print(f"[Gen{generation}]")
print(f"train_dir: {train_dir}")
print(f"eval_dir: {eval_dir}")
print(f"train_status: {train_status}")
print(f"eval_status: {eval_status}")
print("[config]")
print(f"mode: {cfg.get('mode', '')}")
print(f"model.type: {model_cfg.get('type', '')}")
print(f"dataset.use_eval_split: {data_cfg.get('use_eval_split', '')}")
print(f"dataset.eval_ratio: {data_cfg.get('eval_ratio', '')}")
print(f"dataset.split_seed: {data_cfg.get('split_seed', '')}")
print(f"training.batch_size: {tr_cfg.get('batch_size', '')}")
print(f"training.lr: {tr_cfg.get('lr', '')}")
print(f"distillation.temperature: {kd_cfg.get('temperature', '')}")
print(f"distillation.alpha: {kd_cfg.get('alpha', '')}")
print(f"distillation.teacher_checkpoint: {kd_cfg.get('teacher_checkpoint', '')}")
print("[train results]")
print(f"best_eval_acc: {coalesce(train, 'best_eval_acc', 'best_acc')}")
print(f"best_epoch: {coalesce(train, 'best_epoch')}")
print(f"final_train_acc: {coalesce(train, 'final_train_acc')}")
print(f"final_eval_acc: {coalesce(train, 'final_eval_acc', 'final_test_acc')}")
print("[eval results]")
print(f"checkpoint: {ev.get('checkpoint', checkpoint)}")
print(f"top1: {ev.get('top1', 'n/a')}")
print(f"top5: {ev.get('top5', 'n/a')}")
print("")
PY
}

{
    echo "Born Again Distillation Pipeline"
    echo "============================================================"
    echo "job_tag: slurm-train-eval-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo "strategy: Student_0 (from 4201) -> Student_1 -> Student_2 -> Student_3"
    echo ""
} > "$SUMMARY_FILE"

# Configuration used for all generations
# Use the Born Again config which has student as teacher (not slow_r50)
DISTILLATION_CONFIG="experiments/configs/born_again_distillation_t8_a07_24f_lightaug.yaml"

# ============================================================================
# GENERATION 1: Student 1 distillates from 4201's Student (Gen 0)
# ============================================================================
GEN1_PHASE="born_again_gen1"
GEN1_TRAIN_DIR="$ROOT_DIR/$GEN1_PHASE/train"
GEN1_EVAL_DIR="$ROOT_DIR/$GEN1_PHASE/eval"
GEN1_CKPT_DIR="$CHECKPOINT_ROOT/$GEN1_PHASE"
GEN1_CKPT="$GEN1_CKPT_DIR/distillation_best.pth"

echo ""
echo "======== GENERATION 1: Student 1 learns from 4201's Student ========"

run_born_again_training \
    "1" \
    "$DISTILLATION_CONFIG" \
    "$INITIAL_TEACHER_LOCAL" \
    "$GEN1_TRAIN_DIR" \
    "$GEN1_CKPT_DIR" \
    "distillation_best.pth"

if [ -f "$GEN1_CKPT" ]; then
    run_evaluation "1" "$DISTILLATION_CONFIG" "$GEN1_CKPT" "$GEN1_EVAL_DIR"
else
    mkdir -p "$GEN1_EVAL_DIR"
    touch "$GEN1_EVAL_DIR/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$GEN1_EVAL_DIR/status.txt"
    overall_status="FAILED"
fi

append_summary "1" "$GEN1_TRAIN_DIR" "$GEN1_EVAL_DIR" "$GEN1_CKPT"

# ============================================================================
# GENERATION 2: Student 2 distillates from Student 1 (Gen 1)
# ============================================================================
GEN2_PHASE="born_again_gen2"
GEN2_TRAIN_DIR="$ROOT_DIR/$GEN2_PHASE/train"
GEN2_EVAL_DIR="$ROOT_DIR/$GEN2_PHASE/eval"
GEN2_CKPT_DIR="$CHECKPOINT_ROOT/$GEN2_PHASE"
GEN2_CKPT="$GEN2_CKPT_DIR/distillation_best.pth"

echo ""
echo "======== GENERATION 2: Student 2 learns from Student 1 ========"

if [ "$overall_status" = "SUCCESS" ] && [ -f "$GEN1_CKPT" ]; then
    run_born_again_training \
        "2" \
        "$DISTILLATION_CONFIG" \
        "$GEN1_CKPT" \
        "$GEN2_TRAIN_DIR" \
        "$GEN2_CKPT_DIR" \
        "distillation_best.pth"

    if [ -f "$GEN2_CKPT" ]; then
        run_evaluation "2" "$DISTILLATION_CONFIG" "$GEN2_CKPT" "$GEN2_EVAL_DIR"
    else
        mkdir -p "$GEN2_EVAL_DIR"
        touch "$GEN2_EVAL_DIR/status_FAILED"
        printf 'exit_code=%s\n' 99 > "$GEN2_EVAL_DIR/status.txt"
        overall_status="FAILED"
    fi
else
    echo "WARNING: Skipping Generation 2 due to Generation 1 failure"
    mkdir -p "$GEN2_TRAIN_DIR" "$GEN2_EVAL_DIR"
    touch "$GEN2_TRAIN_DIR/status_FAILED"
    touch "$GEN2_EVAL_DIR/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$GEN2_TRAIN_DIR/status.txt"
    printf 'exit_code=%s\n' 99 > "$GEN2_EVAL_DIR/status.txt"
    overall_status="FAILED"
fi

append_summary "2" "$GEN2_TRAIN_DIR" "$GEN2_EVAL_DIR" "$GEN2_CKPT"

# ============================================================================
# GENERATION 3: Student 3 distillates from Student 2 (Gen 2)
# ============================================================================
GEN3_PHASE="born_again_gen3"
GEN3_TRAIN_DIR="$ROOT_DIR/$GEN3_PHASE/train"
GEN3_EVAL_DIR="$ROOT_DIR/$GEN3_PHASE/eval"
GEN3_CKPT_DIR="$CHECKPOINT_ROOT/$GEN3_PHASE"
GEN3_CKPT="$GEN3_CKPT_DIR/distillation_best.pth"

echo ""
echo "======== GENERATION 3: Student 3 learns from Student 2 ========"

if [ "$overall_status" = "SUCCESS" ] && [ -f "$GEN2_CKPT" ]; then
    run_born_again_training \
        "3" \
        "$DISTILLATION_CONFIG" \
        "$GEN2_CKPT" \
        "$GEN3_TRAIN_DIR" \
        "$GEN3_CKPT_DIR" \
        "distillation_best.pth"

    if [ -f "$GEN3_CKPT" ]; then
        run_evaluation "3" "$DISTILLATION_CONFIG" "$GEN3_CKPT" "$GEN3_EVAL_DIR"
    else
        mkdir -p "$GEN3_EVAL_DIR"
        touch "$GEN3_EVAL_DIR/status_FAILED"
        printf 'exit_code=%s\n' 99 > "$GEN3_EVAL_DIR/status.txt"
        overall_status="FAILED"
    fi
else
    echo "WARNING: Skipping Generation 3 due to Generation 2 failure"
    mkdir -p "$GEN3_TRAIN_DIR" "$GEN3_EVAL_DIR"
    touch "$GEN3_TRAIN_DIR/status_FAILED"
    touch "$GEN3_EVAL_DIR/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$GEN3_TRAIN_DIR/status.txt"
    printf 'exit_code=%s\n' 99 > "$GEN3_EVAL_DIR/status.txt"
    overall_status="FAILED"
fi

append_summary "3" "$GEN3_TRAIN_DIR" "$GEN3_EVAL_DIR" "$GEN3_CKPT"

# ============================================================================
# Final Summary
# ============================================================================
{
    echo ""
    echo "Pipeline completed."
    echo "Overall status: $overall_status"
} >> "$SUMMARY_FILE"

echo ""
echo "Pipeline Summary written to: $SUMMARY_FILE"
echo "Overall status: $overall_status"
echo ""
echo "Checkpoints saved in: $CHECKPOINT_ROOT"
echo "  - Gen 1: $GEN1_CKPT_DIR"
echo "  - Gen 2: $GEN2_CKPT_DIR"
echo "  - Gen 3: $GEN3_CKPT_DIR"

if [ "$overall_status" = "SUCCESS" ]; then
    exit 0
else
    exit 1
fi
