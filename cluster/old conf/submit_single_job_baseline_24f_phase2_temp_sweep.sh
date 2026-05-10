#!/bin/bash
# ============================================================================
# Single-job sweep: first 3 baseline runs (24 frames, temperature sweep study)
# Phase 2: Baselines for T8, T10, T12 temperature variants
#
# Usage:
#   sbatch cluster/submit_single_job_baseline_24f_phase2_temp_sweep.sh
# ============================================================================

#SBATCH --job-name=baseline-24f-temp-sweep-p2
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-baseline-24f-phase2-%j.log

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"

if [ ! -d "$PROJECT_DIR" ]; then
    echo "Project directory not found: $PROJECT_DIR"
    exit 1
fi

cd "$PROJECT_DIR"
mkdir -p logs

ROOT_DIR="$PROJECT_DIR/experiments/logs/slurm-baseline-24f-phase2-${SLURM_JOB_ID:-local}"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/slurm-baseline-24f-phase2-${SLURM_JOB_ID:-local}"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT"

overall_status="SUCCESS"

run_training() {
    local run_name="$1"
    local config="$2"
    local train_dir="$3"
    local train_overrides="$4"

    mkdir -p "$train_dir"
    echo ""
    echo ">>> TRAINING: $run_name"
    echo "    config: $config"

    local -a ov=()
    if [ -n "$train_overrides" ]; then
        read -r -a ov <<< "$train_overrides"
    fi

    set +e
    TRAIN_LOG_DIR="$train_dir" \
    TRAIN_CONFIG_PATH="$config" \
    TRAINING_TYPE="$run_name" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        --env PYTHONUNBUFFERED=1 \
        /shared/sifs/latest.sif \
        python -u -m src.training.train --config "$config" --override "${ov[@]}"
    local exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ]; then
        touch "$train_dir/status_SUCCESS"
    else
        touch "$train_dir/status_FAILED"
        overall_status="FAILED"
    fi
    printf 'exit_code=%s\n' "$exit_code" > "$train_dir/status.txt"

    return "$exit_code"
}

run_evaluation() {
    local run_name="$1"
    local config="$2"
    local checkpoint="$3"
    local eval_dir="$4"

    mkdir -p "$eval_dir"
    echo ""
    echo ">>> EVALUATING: $run_name"
    echo "    checkpoint: $checkpoint"

    set +e
    EVAL_LOG_DIR="$eval_dir" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        ${HF_TOKEN:+--env HF_TOKEN="$HF_TOKEN"} \
        --env PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8 \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.evaluate --config "$config" --override "evaluation.checkpoint=$checkpoint"
    local exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ]; then
        touch "$eval_dir/status_SUCCESS"
    else
        touch "$eval_dir/status_FAILED"
        overall_status="FAILED"
    fi
    printf 'exit_code=%s\n' "$exit_code" > "$eval_dir/status.txt"

    return "$exit_code"
}

append_summary() {
    local run_name="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local checkpoint="$4"

    local train_status="UNKNOWN"
    local eval_status="UNKNOWN"
    [ -f "$train_dir/status_SUCCESS" ] && train_status="SUCCESS"
    [ -f "$train_dir/status_FAILED" ] && train_status="FAILED"
    [ -f "$eval_dir/status_SUCCESS" ] && eval_status="SUCCESS"
    [ -f "$eval_dir/status_FAILED" ] && eval_status="FAILED"

    python3 - "$run_name" "$train_dir" "$eval_dir" "$checkpoint" "$train_status" "$eval_status" >> "$SUMMARY_FILE" <<'PY'
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


run_name = sys.argv[1]
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

print(f"[{run_name}]")
print(f"train_dir: {train_dir}")
print(f"eval_dir: {eval_dir}")
print(f"train_status: {train_status}")
print(f"eval_status: {eval_status}")
print("[config]")
print(f"mode: {cfg.get('mode', '')}")
print(f"model.type: {model_cfg.get('type', '')}")
print(f"dataset.num_frames: {data_cfg.get('num_frames', '')}")
print(f"dataset.max_temporal_stride: {data_cfg.get('max_temporal_stride', '')}")
print(f"training.batch_size: {tr_cfg.get('batch_size', '')}")
print(f"training.lr: {tr_cfg.get('lr', '')}")
print(f"training.weight_decay: {tr_cfg.get('weight_decay', '')}")
print(f"training.label_smoothing: {tr_cfg.get('label_smoothing', '')}")
print("[train results]")
print(f"best_acc: {train.get('best_acc', 'n/a')}")
print(f"best_epoch: {train.get('best_epoch', 'n/a')}")
print(f"final_train_acc: {train.get('final_train_acc', 'n/a')}")
print(f"final_test_acc: {train.get('final_test_acc', 'n/a')}")
print(f"final_test_top5: {train.get('final_test_top5', 'n/a')}")
print("[eval results]")
print(f"top1: {ev.get('top1', 'n/a')}")
print(f"top5: {ev.get('top5', 'n/a')}")
print("")
PY
}

{
    echo "Pipeline Summary"
    echo "============================================================"
    echo "job_tag: slurm-baseline-24f-phase2-${SLURM_JOB_ID:-local}"
    echo "slurm_job_id: ${SLURM_JOB_ID:-}"
    echo "root_dir: $ROOT_DIR"
    echo "note: Phase 2 baseline training (temperature sweep: T8, T10, T12 with seed=42)"
    echo ""
} > "$SUMMARY_FILE"

run_one() {
    local run_name="$1"
    local cfg="$2"

    local train_dir="$ROOT_DIR/$run_name/train"
    local eval_dir="$ROOT_DIR/$run_name/eval"
    local ckpt_dir="$CHECKPOINT_ROOT/$run_name"
    local ckpt_path="$ckpt_dir/baseline_best.pth"

    run_training "$run_name" "$cfg" "$train_dir" \
        "training.checkpoint_dir=$ckpt_dir logging.run_name=$run_name"

    if [ -f "$ckpt_path" ]; then
        run_evaluation "$run_name" "$cfg" "$ckpt_path" "$eval_dir"
    else
        overall_status="FAILED"
        mkdir -p "$eval_dir"
        touch "$eval_dir/status_FAILED"
        printf 'exit_code=%s\n' 99 > "$eval_dir/status.txt"
    fi

    append_summary "$run_name" "$train_dir" "$eval_dir" "$ckpt_path"
}

run_one "baseline_t8_24f_lightaug" "experiments/configs/baseline_t8_24f_lightaug.yaml"
run_one "baseline_t10_24f_lightaug" "experiments/configs/baseline_t10_24f_lightaug.yaml"
run_one "baseline_t12_24f_lightaug" "experiments/configs/baseline_t12_24f_lightaug.yaml"

echo ""
echo "Baseline Phase 2 sweep completed in one SLURM job."
echo "Summary: $SUMMARY_FILE"
if [ "$overall_status" = "SUCCESS" ]; then
    echo "Overall status: SUCCESS"
else
    echo "Overall status: FAILED"
    exit 1
fi
