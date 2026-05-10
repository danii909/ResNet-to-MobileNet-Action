#!/bin/bash
# =============================================================================
# submit_at_temporal.sh  —  Run Temporal AT experiments (Symmetric + Heavy)
#
# Settings match slurm-train-eval-4201 EXACTLY (60 ep, cosine, no label
# smoothing, no extra augmentation). Only AT parameters are added.
#
# Runs:
#   1. KD + AT Symmetric  (beta_s=0.05, beta_t=0.05)  → train + eval
#   2. KD + AT Heavy      (beta_s=0.03, beta_t=0.07)  → train + eval
#
# Usage:
#   export TEACHER_CKPT="/home/.../teacher_finetune_best.pth"
#   sbatch cluster/submit_at_temporal.sh
# =============================================================================

#SBATCH --job-name=kd-at-temporal
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-at-temporal-%j.log

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi
cd "$PROJECT_DIR"
mkdir -p logs

# ---------------------------------------------------------------------------
# Reuse teacher from slurm-train-eval-4384 (most recent run).
# ---------------------------------------------------------------------------
TEACHER_CKPT="${TEACHER_CKPT:-$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4384/teacher/teacher_finetune_best.pth}"

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
    echo "       Set TEACHER_CKPT env variable to the correct path."
    exit 1
fi
echo "Using teacher checkpoint: $TEACHER_CKPT"

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-at-temporal-${SLURM_JOB_ID:-local}"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-at-temporal-${SLURM_JOB_ID:-local}"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT" logs

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8
export PYTHONUNBUFFERED=1

overall_status="SUCCESS"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
    local phase="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local checkpoint="$4"

    local train_status="UNKNOWN"
    local eval_status="UNKNOWN"
    [ -f "$train_dir/status_SUCCESS" ] && train_status="SUCCESS"
    [ -f "$train_dir/status_FAILED"  ] && train_status="FAILED"
    [ -f "$eval_dir/status_SUCCESS"  ] && eval_status="SUCCESS"
    [ -f "$eval_dir/status_FAILED"   ] && eval_status="FAILED"

    python3 - "$phase" "$train_dir" "$eval_dir" "$checkpoint" "$train_status" "$eval_status" >> "$SUMMARY_FILE" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path

def read_json(path):
    if not path.exists(): return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except Exception: return {}

def read_kv(path):
    out = {}
    if not path.exists(): return out
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in raw: continue
        k, v = raw.split(":", 1)
        out[k.strip()] = v.strip()
    return out

def c(d, *keys, default="n/a"):
    for k in keys:
        v = d.get(k)
        if v is not None and str(v).strip(): return str(v)
    return default

phase, train_dir, eval_dir, checkpoint, train_status, eval_status = sys.argv[1:]
train_dir = Path(train_dir); eval_dir = Path(eval_dir)

cfg = read_json(train_dir / "config_summary.json")
train = read_kv(train_dir / "training_summary.txt")
ev = read_json(eval_dir / "evaluation_summary.json")

mcfg = cfg.get("model", {}) or {}
kcfg = cfg.get("distillation", {}) or {}
tr = cfg.get("training", {}) or {}

print(f"[{phase}]")
print(f"train_dir: {train_dir}")
print(f"eval_dir: {eval_dir}")
print(f"train_status: {train_status}")
print(f"eval_status: {eval_status}")
print("[config]")
print(f"mode: {cfg.get('mode','')}")
print(f"model.type: {mcfg.get('type','')}")
print(f"training.epochs: {tr.get('epochs','')}")
print(f"training.batch_size: {tr.get('batch_size','')}")
print(f"training.lr: {tr.get('lr','')}")
print(f"training.scheduler: {tr.get('scheduler','')}")
print(f"distillation.temperature: {kcfg.get('temperature','')}")
print(f"distillation.alpha: {kcfg.get('alpha','')}")
print(f"distillation.at_beta_spatial: {kcfg.get('at_beta_spatial','')}")
print(f"distillation.at_beta_temporal: {kcfg.get('at_beta_temporal','')}")
print("[train results]")
print(f"best_eval_acc: {c(train,'best_eval_acc','best_acc')}")
print(f"best_epoch: {c(train,'best_epoch')}")
print(f"final_train_acc: {c(train,'final_train_acc')}")
print(f"final_eval_acc: {c(train,'final_eval_acc')}")
print(f"final_eval_top5: {c(train,'final_eval_top5')}")
print("[eval results]")
print(f"checkpoint: {ev.get('checkpoint', checkpoint)}")
print(f"top1: {ev.get('top1','n/a')}")
print(f"top5: {ev.get('top5','n/a')}")
print(f"size_mb: {ev.get('size_mb','n/a')}")
print(f"inference_avg_ms: {ev.get('inference_avg_ms','n/a')}")
print("")
PY
}

# ---------------------------------------------------------------------------
# === WRITE SUMMARY HEADER ===
# ---------------------------------------------------------------------------
{
    echo "Pipeline Summary — AT Temporal Experiments"
    echo "============================================================"
    echo "job_tag: slurm-at-temporal-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo "teacher_checkpoint: $TEACHER_CKPT"
    echo ""
    echo "Reference baseline (slurm-train-eval-4201 KD-only): 63.78% best_eval_acc"
    echo ""
} > "$SUMMARY_FILE"

# ---------------------------------------------------------------------------
# 1) KD + AT Symmetric (beta_s=0.05, beta_t=0.05)
# ---------------------------------------------------------------------------
at_sym_cfg="experiments/configs/distillation_at_temporal.yaml"
at_sym_train_dir="$ROOT_DIR/at_sym/train"
at_sym_eval_dir="$ROOT_DIR/at_sym/eval"
at_sym_ckpt_dir="$CHECKPOINT_ROOT/at_sym"
at_sym_ckpt="$at_sym_ckpt_dir/distillation_at_best.pth"

run_training "at_sym" "$at_sym_cfg" "$at_sym_train_dir" \
    "training.checkpoint_dir=$at_sym_ckpt_dir distillation.teacher_checkpoint=$TEACHER_CKPT logging.run_name=kd_at_sym"

if [ -f "$at_sym_ckpt" ]; then
    run_evaluation "at_sym" "$at_sym_cfg" "$at_sym_ckpt" "$at_sym_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$at_sym_eval_dir"
    touch "$at_sym_eval_dir/status_FAILED"
fi
append_summary "at_sym" "$at_sym_train_dir" "$at_sym_eval_dir" "$at_sym_ckpt"

# ---------------------------------------------------------------------------
# 2) KD + AT Temporal Heavy (beta_s=0.03, beta_t=0.07)
# ---------------------------------------------------------------------------
at_heavy_cfg="experiments/configs/distillation_at_temporal_heavy.yaml"
at_heavy_train_dir="$ROOT_DIR/at_heavy/train"
at_heavy_eval_dir="$ROOT_DIR/at_heavy/eval"
at_heavy_ckpt_dir="$CHECKPOINT_ROOT/at_heavy"
at_heavy_ckpt="$at_heavy_ckpt_dir/distillation_at_best.pth"

run_training "at_heavy" "$at_heavy_cfg" "$at_heavy_train_dir" \
    "training.checkpoint_dir=$at_heavy_ckpt_dir distillation.teacher_checkpoint=$TEACHER_CKPT logging.run_name=kd_at_heavy"

if [ -f "$at_heavy_ckpt" ]; then
    run_evaluation "at_heavy" "$at_heavy_cfg" "$at_heavy_ckpt" "$at_heavy_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$at_heavy_eval_dir"
    touch "$at_heavy_eval_dir/status_FAILED"
fi
append_summary "at_heavy" "$at_heavy_train_dir" "$at_heavy_eval_dir" "$at_heavy_ckpt"

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "AT Temporal experiments completed."
echo "Summary: $SUMMARY_FILE"

if [ "$overall_status" = "FAILED" ]; then
    echo "Overall status: FAILED"
    exit 1
fi
echo "Overall status: SUCCESS"
