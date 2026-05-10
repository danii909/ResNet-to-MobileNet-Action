#!/bin/bash
# ============================================================================
# Single SLURM job: 3 Independent Distillation Experiments
#
# Layout:
#   experiments/logs/slurm-train-eval-<JOBID>/
#     exp1_warmup10_t10/train, exp1_warmup10_t10/eval
#     exp2_at_temporal_only/train, exp2_at_temporal_only/eval
#     exp3_at_latestage_semantico/train, exp3_at_latestage_semantico/eval
#     pipeline_summary.txt
# ============================================================================

#SBATCH --job-name=kd-3-experiments
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
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
print(f"training.kd_warmup_epochs: {tr_cfg.get('kd_warmup_epochs', '')}")
print(f"distillation.temperature: {kd_cfg.get('temperature', '')}")
print(f"distillation.alpha: {kd_cfg.get('alpha', '')}")
print(f"distillation.at_beta_spatial: {kd_cfg.get('at_beta_spatial', '')}")
print(f"distillation.at_beta_temporal: {kd_cfg.get('at_beta_temporal', '')}")
print(f"distillation.teacher_keys: {kd_cfg.get('teacher_keys', '')}")
print(f"distillation.student_keys: {kd_cfg.get('student_keys', '')}")
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
    echo "Pipeline Summary - 3 KD Experiments"
    echo "============================================================"
    echo "job_tag: slurm-train-eval-${SLURM_JOB_ID:-local}"
    echo "root_dir: $ROOT_DIR"
    echo ""
} > "$SUMMARY_FILE"

# # --- Experiment 1: KD Warmup 10, T=10 ---
# exp1_cfg="experiments/configs/exp1_kd_warmup10_t10.yaml"
# exp1_train_dir="$ROOT_DIR/exp1_warmup10_t10/train"
# exp1_eval_dir="$ROOT_DIR/exp1_warmup10_t10/eval"
# exp1_ckpt_dir="$CHECKPOINT_ROOT/exp1_warmup10_t10"
# exp1_ckpt="$exp1_ckpt_dir/distillation_best.pth"
# 
# run_training "exp1_warmup10_t10" "$exp1_cfg" "$exp1_train_dir" \
#     "training.checkpoint_dir=$exp1_ckpt_dir"
# 
# if [ -f "$exp1_ckpt" ]; then
#     run_evaluation "exp1_warmup10_t10" "$exp1_cfg" "$exp1_ckpt" "$exp1_eval_dir"
# else
#     mkdir -p "$exp1_eval_dir"
#     touch "$exp1_eval_dir/status_FAILED"
#     printf 'exit_code=%s\n' 99 > "$exp1_eval_dir/status.txt"
# fi
# 
# append_summary "exp1_warmup10_t10" "$exp1_train_dir" "$exp1_eval_dir" "$exp1_ckpt"

# --- Experiment 2: AT Temporal Only ---
exp2_cfg="experiments/configs/exp2_at_temporal_only.yaml"
exp2_train_dir="$ROOT_DIR/exp2_at_temporal_only/train"
exp2_eval_dir="$ROOT_DIR/exp2_at_temporal_only/eval"
exp2_ckpt_dir="$CHECKPOINT_ROOT/exp2_at_temporal_only"
exp2_ckpt="$exp2_ckpt_dir/distillation_best.pth"

run_training "exp2_at_temporal_only" "$exp2_cfg" "$exp2_train_dir" \
    "training.checkpoint_dir=$exp2_ckpt_dir"

if [ -f "$exp2_ckpt" ]; then
    run_evaluation "exp2_at_temporal_only" "$exp2_cfg" "$exp2_ckpt" "$exp2_eval_dir"
else
    mkdir -p "$exp2_eval_dir"
    touch "$exp2_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$exp2_eval_dir/status.txt"
fi

append_summary "exp2_at_temporal_only" "$exp2_train_dir" "$exp2_eval_dir" "$exp2_ckpt"

# --- Experiment 3: Late-Stage AT Semantico ---
exp3_cfg="experiments/configs/exp3_at_latestage_semantico.yaml"
exp3_train_dir="$ROOT_DIR/exp3_at_latestage_semantico/train"
exp3_eval_dir="$ROOT_DIR/exp3_at_latestage_semantico/eval"
exp3_ckpt_dir="$CHECKPOINT_ROOT/exp3_at_latestage_semantico"
exp3_ckpt="$exp3_ckpt_dir/distillation_best.pth"

run_training "exp3_at_latestage_semantico" "$exp3_cfg" "$exp3_train_dir" \
    "training.checkpoint_dir=$exp3_ckpt_dir"

if [ -f "$exp3_ckpt" ]; then
    run_evaluation "exp3_at_latestage_semantico" "$exp3_cfg" "$exp3_ckpt" "$exp3_eval_dir"
else
    mkdir -p "$exp3_eval_dir"
    touch "$exp3_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$exp3_eval_dir/status.txt"
fi

append_summary "exp3_at_latestage_semantico" "$exp3_train_dir" "$exp3_eval_dir" "$exp3_ckpt"

echo ""
echo "Pipeline completed."
if [ "$overall_status" = "FAILED" ]; then exit 1; fi
