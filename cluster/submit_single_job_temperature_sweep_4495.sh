#!/bin/bash
# ============================================================================
# Single SLURM job: temperature sweep for KD on the 4495 split.
#
# Runs four distillation trainings with T = 1, 5, 10, 20 while keeping the
# rest of the 4495 setup unchanged. After each run it evaluates the checkpoint
# and generates one t-SNE plot with the fixed teacher and baseline checkpoints.
# ============================================================================

#SBATCH --job-name=kd-temp-sweep-4495
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm-kd-temp-sweep-4495-%j.log

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
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
TSNE_OUTPUT_DIR="$ROOT_DIR/tsne_plots"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT" "$TSNE_OUTPUT_DIR"

TEACHER_CKPT="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth"
BASELINE_CKPT="$PROJECT_DIR/experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth"

BASE_CONFIG="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
NUM_CLASSES=10
ALPHA=0.7
WARMUP_EPOCHS=5
TEMPERATURES=(1 5 10 20)

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

run_tsne() {
    local phase="$1"
    local config="$2"
    local student_ckpt="$3"
    local output_filename="$4"

    echo ""
    echo ">>> t-SNE: $phase"
    echo "    student: $student_ckpt"

    set +e
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.tsne_visualizer \
            --config "$config" \
            --teacher-ckpt "$TEACHER_CKPT" \
            --baseline-ckpt "$BASELINE_CKPT" \
            --student-ckpt "$student_ckpt" \
            --num-classes "$NUM_CLASSES" \
            --output-dir "$TSNE_OUTPUT_DIR" \
            --output-filename "$output_filename"
    local rc=$?
    set -e

    if [ "$rc" -ne 0 ]; then
        overall_status="FAILED"
    fi

    return "$rc"
}

append_summary() {
    local phase="$1"
    local temperature="$2"
    local train_dir="$3"
    local eval_dir="$4"
    local checkpoint="$5"
    local tsne_file="$6"

    local train_status="UNKNOWN"
    local eval_status="UNKNOWN"
    [ -f "$train_dir/status_SUCCESS" ] && train_status="SUCCESS"
    [ -f "$train_dir/status_FAILED" ] && train_status="FAILED"
    [ -f "$eval_dir/status_SUCCESS" ] && eval_status="SUCCESS"
    [ -f "$eval_dir/status_FAILED" ] && eval_status="FAILED"

    python3 - "$phase" "$temperature" "$train_dir" "$eval_dir" "$checkpoint" "$tsne_file" "$train_status" "$eval_status" >> "$SUMMARY_FILE" <<'PY'
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
temperature = sys.argv[2]
train_dir = Path(sys.argv[3])
eval_dir = Path(sys.argv[4])
checkpoint = sys.argv[5]
tsne_file = sys.argv[6]
train_status = sys.argv[7]
eval_status = sys.argv[8]

cfg = read_json(train_dir / "config_summary.json")
train = read_colon_kv(train_dir / "training_summary.txt")
ev = read_json(eval_dir / "evaluation_summary.json")

model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model"), dict) else {}
tr_cfg = cfg.get("training", {}) if isinstance(cfg.get("training"), dict) else {}
kd_cfg = cfg.get("distillation", {}) if isinstance(cfg.get("distillation"), dict) else {}

print(f"[{phase}]")
print(f"temperature: {temperature}")
print(f"train_dir: {train_dir}")
print(f"eval_dir: {eval_dir}")
print(f"train_status: {train_status}")
print(f"eval_status: {eval_status}")
print(f"tsne_plot: {tsne_file}")
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
    echo "Temperature Sweep Summary"
    echo "============================================================"
    echo "job_tag: slurm-train-eval-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo "teacher_ckpt: $TEACHER_CKPT"
    echo "baseline_ckpt: $BASELINE_CKPT"
    echo "base_config: $BASE_CONFIG"
    echo "temperatures: ${TEMPERATURES[*]}"
    echo ""
} > "$SUMMARY_FILE"

for temperature in "${TEMPERATURES[@]}"; do
    phase="kd_t${temperature}_a07_24f_lightaug"
    train_dir="$ROOT_DIR/$phase/train"
    eval_dir="$ROOT_DIR/$phase/eval"
    ckpt_dir="$CHECKPOINT_ROOT/$phase"
    ckpt="$ckpt_dir/distillation_best.pth"
    tsne_file="tsne_temperature_t${temperature}"

    overrides="dataset.use_eval_split=true dataset.eval_ratio=0.2 dataset.split_seed=42 dataset.num_frames=24 dataset.crop_size=112 dataset.resize_short_side=128 dataset.use_random_resized_crop=false dataset.color_jitter_strength=0.1 dataset.random_erasing_prob=0.05 dataset.max_temporal_stride=1 training.mode=distillation training.batch_size=16 training.optimizer=adamw training.lr=0.0005 training.weight_decay=0.01 training.scheduler=cosine training.grad_clip=1.0 training.mixed_precision=true training.label_smoothing=0.05 training.kd_warmup_epochs=$WARMUP_EPOCHS distillation.teacher_checkpoint=$TEACHER_CKPT distillation.temperature=$temperature distillation.alpha=$ALPHA training.checkpoint_dir=$ckpt_dir logging.run_name=$phase"

    run_training "$phase" "$BASE_CONFIG" "$train_dir" "$overrides" || true

    if [ -f "$ckpt" ]; then
        run_evaluation "$phase" "$BASE_CONFIG" "$ckpt" "$eval_dir" || true
        run_tsne "$phase" "$BASE_CONFIG" "$ckpt" "$tsne_file" || true
    else
        overall_status="FAILED"
        mkdir -p "$eval_dir"
        touch "$eval_dir/status_FAILED"
        printf 'exit_code=%s\n' 99 > "$eval_dir/status.txt"
        echo "WARNING: missing checkpoint for $phase, skipping eval and t-SNE."
    fi

    append_summary "$phase" "$temperature" "$train_dir" "$eval_dir" "$ckpt" "$tsne_file"
done

echo ""
echo "Pipeline completed in one SLURM job."
echo "Summary: $SUMMARY_FILE"

if [ "$overall_status" = "FAILED" ]; then
    echo "Overall status: FAILED"
    exit 1
fi

echo "Overall status: SUCCESS"