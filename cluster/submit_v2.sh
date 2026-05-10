#!/bin/bash
# =============================================================================
# submit_v2.sh  —  Improved pipeline: baseline v2 → KD v2 → KD+AT v2 → analysis
#
# Changes vs submit_single_job_train_eval_24f.sh:
#   - SKIPS teacher training: reuses the checkpoint from slurm-train-eval-4201
#   - Runs KD standard v2  (75 ep, temporal stride 2, cosine_warmup)
#   - Runs KD + Attention Transfer v2  (extra objective)
#   - Runs t-SNE visualization on teacher + baseline + KD + KD+AT embeddings
#   - Runs confusion matrix analysis on all student models
#
# Usage:
#   export TEACHER_CKPT="/home/.../experiments/checkpoints/slurm-train-eval-4201/teacher/teacher_finetune_best.pth"
#   sbatch cluster/submit_v2.sh
# =============================================================================

#SBATCH --job-name=kd-v2-pipeline
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-v2-%j.log

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: Project directory not found: $PROJECT_DIR"
    exit 1
fi
cd "$PROJECT_DIR"
mkdir -p logs

# ---------------------------------------------------------------------------
# Reuse teacher from the previous best run (slurm-train-eval-4201).
# Override via env if needed.
# ---------------------------------------------------------------------------
TEACHER_CKPT="${TEACHER_CKPT:-$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4201/teacher/teacher_finetune_best.pth}"

if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: Teacher checkpoint not found: $TEACHER_CKPT"
    echo "       Set TEACHER_CKPT env variable to the correct path."
    exit 1
fi
echo "Using teacher checkpoint: $TEACHER_CKPT"

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-train-eval-${SLURM_JOB_ID:-local}"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-${SLURM_JOB_ID:-local}"
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

run_confusion() {
    local label="$1"
    local model_type="$2"
    local config="$3"
    local checkpoint="$4"
    local confusion_dir="$5"

    echo ""
    echo ">>> CONFUSION MATRIX: $label"

    set +e
    CONFUSION_LOG_DIR="$confusion_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.confusion \
            --config "$config" \
            --override \
                "model.type=$model_type" \
                "evaluation.checkpoint=$checkpoint" \
                "evaluation.label=$label" \
                "evaluation.confusion_dir=$confusion_dir"
    local rc=$?
    set -e

    if [ "$rc" -eq 0 ]; then
        touch "$confusion_dir/status_SUCCESS_${label}"
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
# Pipeline Summary helpers
# ---------------------------------------------------------------------------
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
print(f"training.batch_size: {tr.get('batch_size','')}")
print(f"training.lr: {tr.get('lr','')}")
print(f"training.scheduler: {tr.get('scheduler','')}")
print(f"training.label_smoothing: {tr.get('label_smoothing','')}")
print(f"distillation.teacher_checkpoint: {kcfg.get('teacher_checkpoint','')}")
print(f"distillation.temperature: {kcfg.get('temperature','')}")
print(f"distillation.alpha: {kcfg.get('alpha','')}")
print(f"distillation.at_beta: {kcfg.get('at_beta','')}")
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
    echo "Pipeline Summary v2"
    echo "============================================================"
    echo "job_tag: slurm-train-eval-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo "teacher_checkpoint: $TEACHER_CKPT"
    echo ""
} > "$SUMMARY_FILE"

# Common overrides shared by all student runs
COMMON_OVERRIDES="dataset.use_eval_split=true dataset.eval_ratio=0.2 dataset.split_seed=42 dataset.num_frames=24 dataset.crop_size=112 dataset.resize_short_side=128 dataset.use_random_resized_crop=false dataset.color_jitter_strength=0.1 dataset.random_erasing_prob=0.05 dataset.max_temporal_stride=2 training.batch_size=16 training.optimizer=adamw training.lr=0.0005 training.weight_decay=0.01 training.scheduler=cosine_warmup training.warmup_epochs=5 training.warmup_start_factor=0.1 training.grad_clip=1.0 training.mixed_precision=true training.epochs=75 training.kd_warmup_epochs=3"

# ---------------------------------------------------------------------------
# 1) BASELINE v2
# ---------------------------------------------------------------------------
baseline_cfg="experiments/configs/baseline_24f_v2.yaml"
baseline_train_dir="$ROOT_DIR/baseline/train"
baseline_eval_dir="$ROOT_DIR/baseline/eval"
baseline_ckpt_dir="$CHECKPOINT_ROOT/baseline"
baseline_ckpt="$baseline_ckpt_dir/baseline_best.pth"

run_training "baseline" "$baseline_cfg" "$baseline_train_dir" \
    "$COMMON_OVERRIDES training.mode=baseline training.label_smoothing=0.1 training.checkpoint_dir=$baseline_ckpt_dir logging.run_name=baseline_24f_v2"

if [ -f "$baseline_ckpt" ]; then
    run_evaluation "baseline" "$baseline_cfg" "$baseline_ckpt" "$baseline_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$baseline_eval_dir"
    touch "$baseline_eval_dir/status_FAILED"
fi
append_summary "baseline" "$baseline_train_dir" "$baseline_eval_dir" "$baseline_ckpt"

# ---------------------------------------------------------------------------
# 2) KD STANDARD v2
# ---------------------------------------------------------------------------
kd_cfg="experiments/configs/distillation_t8_a07_24f_v2.yaml"
kd_train_dir="$ROOT_DIR/distillation/train"
kd_eval_dir="$ROOT_DIR/distillation/eval"
kd_ckpt_dir="$CHECKPOINT_ROOT/distillation"
kd_ckpt="$kd_ckpt_dir/distillation_best.pth"

run_training "distillation" "$kd_cfg" "$kd_train_dir" \
    "$COMMON_OVERRIDES training.mode=distillation training.label_smoothing=0.05 training.checkpoint_dir=$kd_ckpt_dir distillation.teacher_checkpoint=$TEACHER_CKPT distillation.temperature=8.0 distillation.alpha=0.7 logging.run_name=kd_t8_a07_24f_v2"

if [ -f "$kd_ckpt" ]; then
    run_evaluation "distillation" "$kd_cfg" "$kd_ckpt" "$kd_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$kd_eval_dir"
    touch "$kd_eval_dir/status_FAILED"
fi
append_summary "distillation" "$kd_train_dir" "$kd_eval_dir" "$kd_ckpt"

# ---------------------------------------------------------------------------
# 3) KD + SPATIAL & TEMPORAL ATTENTION TRANSFER  (Extra Objective)
# ---------------------------------------------------------------------------
at_cfg="experiments/configs/distillation_at_temporal.yaml"
at_train_dir="$ROOT_DIR/distillation_at/train"
at_eval_dir="$ROOT_DIR/distillation_at/eval"
at_ckpt_dir="$CHECKPOINT_ROOT/distillation_at"
at_ckpt="$at_ckpt_dir/distillation_at_best.pth"

run_training "distillation_at" "$at_cfg" "$at_train_dir" \
    "$COMMON_OVERRIDES training.mode=distillation_at training.label_smoothing=0.05 training.checkpoint_dir=$at_ckpt_dir distillation.teacher_checkpoint=$TEACHER_CKPT distillation.temperature=8.0 distillation.alpha=0.7 distillation.at_beta_spatial=0.05 distillation.at_beta_temporal=0.05 logging.run_name=kd_at_spatial_temporal_sym_24f"

if [ -f "$at_ckpt" ]; then
    run_evaluation "distillation_at" "$at_cfg" "$at_ckpt" "$at_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$at_eval_dir"
    touch "$at_eval_dir/status_FAILED"
fi
append_summary "distillation_at" "$at_train_dir" "$at_eval_dir" "$at_ckpt"

# ---------------------------------------------------------------------------
# 4) ANALYSIS: Confusion Matrix  (Extra Objective)
# ---------------------------------------------------------------------------
echo ""
echo ">>> POST-TRAINING ANALYSIS"
confusion_dir="$ROOT_DIR/analysis/confusion"
mkdir -p "$confusion_dir"

# Teacher (already trained, use existing checkpoint)
run_confusion "teacher" "teacher" "$kd_cfg" "$TEACHER_CKPT" "$confusion_dir" || true

# Baseline v2
if [ -f "$baseline_ckpt" ]; then
    run_confusion "baseline_v2" "student" "$baseline_cfg" "$baseline_ckpt" "$confusion_dir" || true
fi

# KD standard v2
if [ -f "$kd_ckpt" ]; then
    run_confusion "distillation_v2" "student" "$kd_cfg" "$kd_ckpt" "$confusion_dir" || true
fi

# KD + AT v2
if [ -f "$at_ckpt" ]; then
    run_confusion "distillation_at_v2" "student" "$at_cfg" "$at_ckpt" "$confusion_dir" || true
fi

# ---------------------------------------------------------------------------
# 5) ANALYSIS: t-SNE Embeddings  (Extra Objective)
# ---------------------------------------------------------------------------
tsne_dir="$ROOT_DIR/analysis/tsne"
mkdir -p "$tsne_dir"

echo ""
echo ">>> t-SNE VISUALIZATION"

# Build override string with all available checkpoints
TSNE_OVERRIDES="evaluation.tsne_dir=$tsne_dir evaluation.teacher_checkpoint=$TEACHER_CKPT"
[ -f "$baseline_ckpt" ] && TSNE_OVERRIDES="$TSNE_OVERRIDES evaluation.baseline_checkpoint=$baseline_ckpt"
[ -f "$kd_ckpt"       ] && TSNE_OVERRIDES="$TSNE_OVERRIDES evaluation.distilled_checkpoint=$kd_ckpt"
[ -f "$at_ckpt"       ] && TSNE_OVERRIDES="$TSNE_OVERRIDES evaluation.at_distilled_checkpoint=$at_ckpt"

set +e
TSNE_LOG_DIR="$tsne_dir" \
apptainer run --nv \
    --env WANDB_MODE=offline \
    --env HF_DATASETS_OFFLINE=1 \
    ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
    --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
    /shared/sifs/latest.sif \
    python -u -m src.evaluation.tsne_visualizer \
        --config "$kd_cfg" \
        --override $TSNE_OVERRIDES
tsne_rc=$?
set -e

if [ "$tsne_rc" -eq 0 ]; then
    echo "t-SNE analysis completed successfully."
    touch "$tsne_dir/status_SUCCESS"
else
    echo "WARNING: t-SNE analysis failed (rc=$tsne_rc) — non-fatal, pipeline continues."
    touch "$tsne_dir/status_FAILED"
fi

# ---------------------------------------------------------------------------
# Final status
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Pipeline v2 completed."
echo "Summary: $SUMMARY_FILE"

if [ "$overall_status" = "FAILED" ]; then
    echo "Overall status: FAILED"
    exit 1
fi
echo "Overall status: SUCCESS"
