#!/bin/bash
# ============================================================================
# Single SLURM job: TAKD Pipeline with slow_r18
# ResNet-18 assistant (KD from ResNet-50) -> MobileNet (KD from assistant). 
# Train + eval for both, 4414-style layout.
# ============================================================================

#SBATCH --job-name=takd-pipeline
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm-takd-pipeline-%j.log

set -euo pipefail

# Avoid "No space left on device" in /tmp
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
print(f"distillation.teacher_type: {kd_cfg.get('teacher_type', '')}")
print(f"distillation.temperature: {kd_cfg.get('temperature', '')}")
print(f"distillation.alpha: {kd_cfg.get('alpha', '')}")
print("[train results]")
print(f"best_eval_acc: {coalesce(train, 'best_eval_acc', 'best_acc')}")
print(f"best_epoch: {coalesce(train, 'best_epoch')}")
print(f"final_train_acc: {coalesce(train, 'final_train_acc')}")
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
    echo "root_dir: $ROOT_DIR"
    echo ""
} > "$SUMMARY_FILE"

# --- Phase 1: Assistant r3d_18 distilled from ResNet-50 ---
teacher_r50_ckpt="/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth"
assistant_cfg="experiments/configs/distillation_takd_assistant_r3d_18.yaml"
assistant_train_dir="$ROOT_DIR/assistant_r3d18/train"
assistant_eval_dir="$ROOT_DIR/assistant_r3d18/eval"
assistant_ckpt_dir="$CHECKPOINT_ROOT/assistant_r3d18"
assistant_ckpt="$assistant_ckpt_dir/distillation_best.pth"

run_training "assistant_r3d18" "$assistant_cfg" "$assistant_train_dir" \
    "training.checkpoint_dir=$assistant_ckpt_dir distillation.teacher_checkpoint=$teacher_r50_ckpt"

if [ -f "$assistant_ckpt" ]; then
    run_evaluation "assistant_r3d18" "$assistant_cfg" "$assistant_ckpt" "$assistant_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$assistant_eval_dir"
    touch "$assistant_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$assistant_eval_dir/status.txt"
fi

append_summary "assistant_r3d18" "$assistant_train_dir" "$assistant_eval_dir" "$assistant_ckpt"

# --- Phase 2: MobileNet distilled from r3d_18 assistant ---
mobilenet_cfg="experiments/configs/distillation_takd_mobilenet_from_r3d_18.yaml"
mobilenet_train_dir="$ROOT_DIR/mobilenet_from_r3d18/train"
mobilenet_eval_dir="$ROOT_DIR/mobilenet_from_r3d18/eval"
mobilenet_ckpt_dir="$CHECKPOINT_ROOT/mobilenet"
mobilenet_ckpt="$mobilenet_ckpt_dir/distillation_best.pth"

run_training "mobilenet_from_r3d18" "$mobilenet_cfg" "$mobilenet_train_dir" \
    "training.checkpoint_dir=$mobilenet_ckpt_dir distillation.teacher_checkpoint=$assistant_ckpt"

if [ -f "$mobilenet_ckpt" ]; then
    run_evaluation "mobilenet_from_r3d18" "$mobilenet_cfg" "$mobilenet_ckpt" "$mobilenet_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$mobilenet_eval_dir"
    touch "$mobilenet_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$mobilenet_eval_dir/status.txt"
fi

append_summary "mobilenet_from_r3d18" "$mobilenet_train_dir" "$mobilenet_eval_dir" "$mobilenet_ckpt"

echo ""
echo "Pipeline completed."
if [ "$overall_status" = "FAILED" ]; then exit 1; fi
