#!/bin/bash
# ============================================================================
# Single SLURM job: teacher -> baseline -> distillation (lightaug 24f), with
# eval after each phase. Re-run of the old refine3-b setup in current conditions.
# ============================================================================

#SBATCH --job-name=repro-4142-lightaug-teacher
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
data_cfg = cfg.get("dataset", {}) if isinstance(cfg.get("dataset"), dict) else {}
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
print(f"dataset.use_eval_split: {data_cfg.get('use_eval_split', '')}")
print(f"dataset.eval_ratio: {data_cfg.get('eval_ratio', '')}")
print(f"dataset.split_seed: {data_cfg.get('split_seed', '')}")
print(f"dataset.max_temporal_stride: {data_cfg.get('max_temporal_stride', '')}")
print(f"training.batch_size: {tr_cfg.get('batch_size', '')}")
print(f"training.lr: {tr_cfg.get('lr', '')}")
print(f"training.weight_decay: {tr_cfg.get('weight_decay', '')}")
print(f"training.label_smoothing: {tr_cfg.get('label_smoothing', '')}")
print(f"training.kd_warmup_epochs: {tr_cfg.get('kd_warmup_epochs', '')}")
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

shared_overrides="dataset.use_eval_split=true dataset.eval_ratio=0.2 dataset.split_seed=42"

# 1) Teacher (lightaug)
teacher_cfg="experiments/configs/teacher_24f_evalsplit.yaml"
teacher_phase="teacher_24f_lightaug"
teacher_train_dir="$ROOT_DIR/$teacher_phase/train"
teacher_eval_dir="$ROOT_DIR/$teacher_phase/eval"
teacher_ckpt_dir="$CHECKPOINT_ROOT/$teacher_phase"
teacher_ckpt="$teacher_ckpt_dir/teacher_finetune_best.pth"

run_training "$teacher_phase" "$teacher_cfg" "$teacher_train_dir" \
    "$shared_overrides training.checkpoint_dir=$teacher_ckpt_dir logging.run_name=$teacher_phase"

if [ -f "$teacher_ckpt" ]; then
    run_evaluation "$teacher_phase" "$teacher_cfg" "$teacher_ckpt" "$teacher_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$teacher_eval_dir"
    touch "$teacher_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$teacher_eval_dir/status.txt"
fi

append_summary "$teacher_phase" "$teacher_train_dir" "$teacher_eval_dir" "$teacher_ckpt"

# 2) Baseline (lightaug)
baseline_cfg="experiments/configs/baseline_ls005_24f_lightaug.yaml"
baseline_phase="baseline_ls005_24f_lightaug"
baseline_train_dir="$ROOT_DIR/$baseline_phase/train"
baseline_eval_dir="$ROOT_DIR/$baseline_phase/eval"
baseline_ckpt_dir="$CHECKPOINT_ROOT/$baseline_phase"
baseline_ckpt="$baseline_ckpt_dir/baseline_best.pth"

run_training "$baseline_phase" "$baseline_cfg" "$baseline_train_dir" \
    "$shared_overrides training.checkpoint_dir=$baseline_ckpt_dir logging.run_name=$baseline_phase"

if [ -f "$baseline_ckpt" ]; then
    run_evaluation "$baseline_phase" "$baseline_cfg" "$baseline_ckpt" "$baseline_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$baseline_eval_dir"
    touch "$baseline_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$baseline_eval_dir/status.txt"
fi

append_summary "$baseline_phase" "$baseline_train_dir" "$baseline_eval_dir" "$baseline_ckpt"

# 3) Distillation (lightaug, T=8, alpha=0.7)
kd_cfg="experiments/configs/distillation_t8_a07_24f_lightaug.yaml"
kd_phase="kd_t8_a07_24f_lightaug"
kd_train_dir="$ROOT_DIR/$kd_phase/train"
kd_eval_dir="$ROOT_DIR/$kd_phase/eval"
kd_ckpt_dir="$CHECKPOINT_ROOT/$kd_phase"
kd_ckpt="$kd_ckpt_dir/distillation_best.pth"

run_training "$kd_phase" "$kd_cfg" "$kd_train_dir" \
    "$shared_overrides training.checkpoint_dir=$kd_ckpt_dir distillation.teacher_checkpoint=$teacher_ckpt logging.run_name=$kd_phase"

if [ -f "$kd_ckpt" ]; then
    run_evaluation "$kd_phase" "$kd_cfg" "$kd_ckpt" "$kd_eval_dir"
else
    overall_status="FAILED"
    mkdir -p "$kd_eval_dir"
    touch "$kd_eval_dir/status_FAILED"
    printf 'exit_code=%s\n' 99 > "$kd_eval_dir/status.txt"
fi

append_summary "$kd_phase" "$kd_train_dir" "$kd_eval_dir" "$kd_ckpt"

# 4) Latent Space t-SNE Visualization
echo ""
echo ">>> GENERATING t-SNE VISUALIZATION"
if [ "$overall_status" != "FAILED" ] && [ -f "$teacher_ckpt" ] && [ -f "$kd_ckpt" ]; then
    tsne_output_dir="$ROOT_DIR/tsne_plots"
    mkdir -p "$tsne_output_dir"
    
    set +e
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
            --baseline-ckpt "$baseline_ckpt" \
            --student-ckpt "$kd_ckpt" \
            --num-classes 10 \
            --output-dir "$tsne_output_dir" \
            --output-filename "tsne_comparison_${SLURM_JOB_ID:-local}"
    tsne_rc=$?
    set -e
    
    if [ "$tsne_rc" -eq 0 ]; then
        echo "t-SNE visualization generated in $tsne_output_dir"
    else
        echo "Warning: t-SNE visualization failed with exit code $tsne_rc"
    fi
else
    echo "Skipping t-SNE visualization because previous steps failed or checkpoints are missing."
fi

echo ""
echo "Pipeline completed in one SLURM job."
echo "Summary: $SUMMARY_FILE"

if [ "$overall_status" = "FAILED" ]; then
    echo "Overall status: FAILED"
    exit 1
fi

echo "Overall status: SUCCESS"
