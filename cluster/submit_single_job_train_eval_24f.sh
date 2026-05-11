#!/bin/bash
# ============================================================================
# Single SLURM job: teacher -> baseline -> distillation, with eval after each.
# Logs layout matches:
#   experiments/logs/slurm-train-eval-<JOBID>/
#     teacher/train, teacher/eval
#     baseline/train, baseline/eval
#     distillation/train, distillation/eval
#     pipeline_summary.txt
# ============================================================================

#SBATCH --job-name=train-eval-24f
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

ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-train-eval-${SLURM_JOB_ID:-local}"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-${SLURM_JOB_ID:-local}"
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
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.evaluate --config "$config" --override "evaluation.checkpoint=$checkpoint"
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
    local phase="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local checkpoint="$4"

    local train_status="UNKNOWN"
    local eval_status="UNKNOWN"
    [ -f "$train_dir/status_SUCCESS" ] && train_status="SUCCESS"
    [ -f "$train_dir/status_FAILED" ] && train_status="FAILED"
    [ -f "$eval_dir/status_SUCCESS" ] && eval_status="SUCCESS"
    [ -f "$eval_dir/status_FAILED" ] && eval_status="FAILED"

    python3 - "$phase" "$train_dir" "$eval_dir" "$checkpoint" "$train_status" "$eval_status" >> "$SUMMARY_FILE" <<'PY'
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


phase = sys.argv[1]
train_dir = Path(sys.argv[2])
eval_dir = Path(sys.argv[3])
checkpoint = sys.argv[4]
train_status = sys.argv[5]
eval_status = sys.argv[6]

cfg = read_json(train_dir / "config_summary.json")
train = read_colon_kv(train_dir / "training_summary.txt")
ev = read_json(eval_dir / "evaluation_summary.json")

model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
tr_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
kd_cfg = cfg.get("distillation", {}) if isinstance(cfg.get("distillation"), dict) else {}

print(f"[{phase}]")
print(f"train_dir: {train_dir}")
print(f"eval_dir: {eval_dir}")
print(f"train_status: {train_status}")
print(f"eval_status: {eval_status}")
print("[config]")
print(f"mode: {cfg.get('mode', '')}")
print(f"model.type: {model_cfg.get('type', '')}")
print(f"training.batch_size: {tr_cfg.get('batch_size', '')}")
print(f"training.lr: {tr_cfg.get('lr', '')}")
print(f"training.weight_decay: {tr_cfg.get('weight_decay', '')}")
print(f"training.label_smoothing: {tr_cfg.get('label_smoothing', '')}")
print(f"distillation.teacher_checkpoint: {kd_cfg.get('teacher_checkpoint', '')}")
print(f"distillation.temperature: {kd_cfg.get('temperature', '')}")
print(f"distillation.alpha: {kd_cfg.get('alpha', '')}")
print("[train results]")
print(f"best_eval_acc: {coalesce(train, 'best_eval_acc', 'best_acc')}")
print(f"best_epoch: {coalesce(train, 'best_epoch')}")
print(f"final_train_acc: {coalesce(train, 'final_train_acc')}")
print(f"final_eval_acc: {coalesce(train, 'final_eval_acc', 'final_test_acc')}")
print(f"final_eval_top5: {coalesce(train, 'final_eval_top5', 'final_test_top5')}")
print("[eval results]")
print(f"checkpoint: {ev.get('checkpoint', checkpoint)}")
print(f"top1: {ev.get('top1', 'n/a')}")
print(f"top5: {ev.get('top5', 'n/a')}")
print("")
PY
}

{
    echo "Pipeline Summary"
    echo "============================================================"
    echo "job_tag: slurm-train-eval-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo ""
} > "$SUMMARY_FILE"

# 1) Teacher first
teacher_cfg="experiments/configs/teacher_24f_evalsplit.yaml"
teacher_train_dir="$ROOT_DIR/teacher/train"
teacher_eval_dir="$ROOT_DIR/teacher/eval"
teacher_ckpt_dir="$CHECKPOINT_ROOT/teacher"
teacher_ckpt="$teacher_ckpt_dir/teacher_finetune_best.pth"

run_training "teacher" "$teacher_cfg" "$teacher_train_dir" \
    "dataset.use_eval_split=true dataset.eval_ratio=0.2 dataset.split_seed=42 training.checkpoint_dir=$teacher_ckpt_dir logging.run_name=teacher_24f_evalsplit"

if [ -f "$teacher_ckpt" ]; then
    run_evaluation "teacher" "$teacher_cfg" "$teacher_ckpt" "$teacher_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$teacher_eval_dir"
    touch "$teacher_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$teacher_eval_dir/status.txt"
fi

append_summary "teacher" "$teacher_train_dir" "$teacher_eval_dir" "$teacher_ckpt"

# Shared non-KD settings for a fair baseline vs distillation comparison.
common_student_overrides="dataset.use_eval_split=true dataset.eval_ratio=0.2 dataset.split_seed=42 dataset.num_frames=24 dataset.crop_size=112 dataset.resize_short_side=128 dataset.use_random_resized_crop=false dataset.color_jitter_strength=0.1 dataset.random_erasing_prob=0.05 dataset.max_temporal_stride=1 training.batch_size=16 training.optimizer=adamw training.lr=0.0005 training.weight_decay=0.01 training.scheduler=cosine training.grad_clip=1.0 training.mixed_precision=true training.label_smoothing=0.05"

# 2) Baseline
baseline_cfg="experiments/configs/baseline_ls005_24f_lightaug.yaml"
baseline_train_dir="$ROOT_DIR/baseline/train"
baseline_eval_dir="$ROOT_DIR/baseline/eval"
baseline_ckpt_dir="$CHECKPOINT_ROOT/baseline"
baseline_ckpt="$baseline_ckpt_dir/baseline_best.pth"

run_training "baseline" "$baseline_cfg" "$baseline_train_dir" \
    "$common_student_overrides training.mode=baseline training.checkpoint_dir=$baseline_ckpt_dir logging.run_name=baseline_ls005_24f_lightaug"

if [ -f "$baseline_ckpt" ]; then
    run_evaluation "baseline" "$baseline_cfg" "$baseline_ckpt" "$baseline_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$baseline_eval_dir"
    touch "$baseline_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$baseline_eval_dir/status.txt"
fi

append_summary "baseline" "$baseline_train_dir" "$baseline_eval_dir" "$baseline_ckpt"

# 3) Distillation (uses teacher from step 1)
kd_cfg="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
kd_train_dir="$ROOT_DIR/distillation/train"
kd_eval_dir="$ROOT_DIR/distillation/eval"
kd_ckpt_dir="$CHECKPOINT_ROOT/distillation"
kd_ckpt="$kd_ckpt_dir/distillation_best.pth"

run_training "distillation" "$kd_cfg" "$kd_train_dir" \
    "$common_student_overrides training.mode=distillation training.kd_warmup_epochs=5 distillation.teacher_checkpoint=$teacher_ckpt distillation.temperature=8.0 distillation.alpha=0.7 training.checkpoint_dir=$kd_ckpt_dir logging.run_name=kd_t8_a07_24f_lightaug"

if [ -f "$kd_ckpt" ]; then
    run_evaluation "distillation" "$kd_cfg" "$kd_ckpt" "$kd_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$kd_eval_dir"
    touch "$kd_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$kd_eval_dir/status.txt"
fi

append_summary "distillation" "$kd_train_dir" "$kd_eval_dir" "$kd_ckpt"

echo ""
echo "Pipeline completed in one SLURM job."
echo "Summary: $SUMMARY_FILE"

if [ "$overall_status" = "FAILED" ]; then
    echo "Overall status: FAILED"
    exit 1
fi

echo "Overall status: SUCCESS"
