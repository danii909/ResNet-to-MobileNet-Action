#!/bin/bash
# ============================================================================
# Born Again Distillation Only Pipeline (Stages 3-5)
# This script runs the sequence of Born Again generations after a crash 
# in the previous stages.
# ============================================================================

#SBATCH --job-name=born-again-only
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-xlarge
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1 --gres=shard:22528
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your@email.com
#SBATCH --output=logs/slurm-born-again-%j.log

set -euo pipefail

# Avoid "No space left on device" in /tmp by using a local tmp directory
mkdir -p "$HOME/tmp"
export TMPDIR="$HOME/tmp"

PROJECT_DIR="${PROJECT_DIR:-$HOME/dl26-projects}"
cd "$PROJECT_DIR"
mkdir -p logs

export WANDB_MODE=offline
export HF_DATASETS_OFFLINE=1
export PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.8

ROOT_JOB_NAME="${ROOT_JOB_NAME:-slurm-born-again-${SLURM_JOB_ID:-local}}"
ROOT_DIR="$PROJECT_DIR/experiments/logs/$ROOT_JOB_NAME"
CHECKPOINT_ROOT="$PROJECT_DIR/experiments/checkpoints/$ROOT_JOB_NAME"
SUMMARY_FILE="$ROOT_DIR/pipeline_summary.txt"
mkdir -p "$ROOT_DIR" "$CHECKPOINT_ROOT"

# Teachers Paths
TEACHER_R50="/home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/slurm-train-eval-4495/teacher_24f_lightaug/teacher_finetune_best.pth"
# The first stage uses a student model (MobileNet) as teacher
TEACHER_STUDENT_BA_START="$HOME/dl26-projects/experiments/checkpoints/slurm-train-eval-4495/kd_t8_a07_24f_lightaug/distillation_best.pth"

overall_status="SUCCESS"

run_stage() {
    local stage_name="$1"
    local config="$2"
    local teacher_ckpt="$3"
    local stage_dir="$ROOT_DIR/$stage_name"
    local stage_ckpt_dir="$CHECKPOINT_ROOT/$stage_name"
    
    mkdir -p "$stage_dir/train" "$stage_dir/eval" "$stage_ckpt_dir"
    
    if [ -f "$stage_dir/train/status_SUCCESS" ]; then
        echo ">>> STAGE $stage_name: Training already SUCCESS."
        local best_ckpt="$stage_ckpt_dir/distillation_at_best.pth"
        [ ! -f "$best_ckpt" ] && best_ckpt="$stage_ckpt_dir/distillation_best.pth"
        [ ! -f "$best_ckpt" ] && best_ckpt="$stage_ckpt_dir/baseline_best.pth"

        if [ ! -f "$stage_dir/eval/status_SUCCESS" ] && [ -f "$best_ckpt" ]; then
            echo "    But evaluation is missing. Running evaluation now..."
            set +e
            EVAL_LOG_DIR="$stage_dir/eval" \
            apptainer run --nv \
                --env WANDB_MODE=offline \
                --env HF_DATASETS_OFFLINE=1 \
                --env TMPDIR="$TMPDIR" \
                /shared/sifs/latest.sif \
                python -u -m src.evaluation.evaluate \
                    --config "$config" \
                    --override "evaluation.checkpoint=$best_ckpt"
            set -e
            touch "$stage_dir/eval/status_SUCCESS"
        fi

        append_to_summary "$stage_name" "$stage_dir/train" "$stage_dir/eval" "$best_ckpt"
        return 0
    fi

    echo ""
    echo ">>> STARTING STAGE: $stage_name"
    echo "    config: $config"
    echo "    teacher: $teacher_ckpt"
    echo "    teacher_type: student"
    
    if [ ! -f "$teacher_ckpt" ]; then
        echo "ERROR: Teacher checkpoint not found: $teacher_ckpt"
        overall_status="FAILED"
        return 1
    fi

    # Training
    set +e
    TRAIN_LOG_DIR="$stage_dir/train" \
    TRAIN_CONFIG_PATH="$config" \
    apptainer run --nv \
        --env WANDB_MODE=offline \
        --env HF_DATASETS_OFFLINE=1 \
        --env PYTHONUNBUFFERED=1 \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.training.train \
            --config "$config" \
            --override \
                "training.checkpoint_dir=$stage_ckpt_dir" \
                "distillation.teacher_checkpoint=$teacher_ckpt" \
                "distillation.teacher_type=student" \
                "logging.run_name=$stage_name"
    local rc=$?
    set -e
    
    if [ "$rc" -eq 0 ]; then
        touch "$stage_dir/train/status_SUCCESS"
    else
        touch "$stage_dir/train/status_FAILED"
        overall_status="FAILED"
        return "$rc"
    fi

    # Evaluation
    local best_ckpt="$stage_ckpt_dir/distillation_at_best.pth"
    [ ! -f "$best_ckpt" ] && best_ckpt="$stage_ckpt_dir/distillation_best.pth"
    [ ! -f "$best_ckpt" ] && best_ckpt="$stage_ckpt_dir/baseline_best.pth"

    if [ -f "$best_ckpt" ]; then
        set +e
        EVAL_LOG_DIR="$stage_dir/eval" \
        apptainer run --nv \
            --env WANDB_MODE=offline \
            --env HF_DATASETS_OFFLINE=1 \
            --env TMPDIR="$TMPDIR" \
            /shared/sifs/latest.sif \
            python -u -m src.evaluation.evaluate \
                --config "$config" \
                --override "evaluation.checkpoint=$best_ckpt"
        set -e
        touch "$stage_dir/eval/status_SUCCESS"
    fi

    # Summary Append
    append_to_summary "$stage_name" "$stage_dir/train" "$stage_dir/eval" "$best_ckpt"
}

append_to_summary() {
    local stage="$1"
    local train_dir="$2"
    local eval_dir="$3"
    local ckpt="$4"
    
    python3 - "$stage" "$train_dir" "$eval_dir" "$ckpt" >> "$SUMMARY_FILE" <<'PY'
import json, sys, re
from pathlib import Path
def read_json(p):
    try: return json.loads(p.read_text()) if p.exists() else {}
    except: return {}
def read_kv(p):
    out = {}
    if p.exists():
        for l in p.read_text().splitlines():
            if ":" in l: k,v = l.split(":",1); out[k.strip()] = v.strip()
    return out
stage, t_dir, e_dir, ckpt = sys.argv[1:5]
cfg = read_json(Path(t_dir)/"config_summary.json")
train = read_kv(Path(t_dir)/"training_summary.txt")
ev = read_json(Path(e_dir)/"evaluation_summary.json")
status = "SUCCESS" if (Path(t_dir)/"status_SUCCESS").exists() else "FAILED/SKIPPED"
print(f"[{stage}]")
print(f"train_status: {status}")
print(f"best_eval_acc: {train.get('best_eval_acc', train.get('best_acc', 'n/a'))}")
print(f"final_train_acc: {train.get('final_train_acc', 'n/a')}")
print(f"test_top1: {ev.get('top1', 'n/a')}")
print(f"teacher: {cfg.get('distillation', {}).get('teacher_checkpoint', 'n/a')}")
print("")
PY
}

# Header Summary
{
    echo "Born Again Distillation Pipeline (Stages 3-5 Only)"
    echo "============================================================"
    echo "job_id: ${SLURM_JOB_ID:-local}"
    echo "root_dir: $ROOT_DIR"
    echo ""
} > "$SUMMARY_FILE"

# --- STAGE 3: Born Again Gen 1 ---
run_stage "born_again_gen1" "experiments/configs/born_again_p50.yaml" "$TEACHER_STUDENT_BA_START" || true

# --- STAGE 4: Born Again Gen 2 ---
GEN1_CKPT="$CHECKPOINT_ROOT/born_again_gen1/distillation_at_best.pth"
[ ! -f "$GEN1_CKPT" ] && GEN1_CKPT="$CHECKPOINT_ROOT/born_again_gen1/distillation_best.pth"

if [ -f "$GEN1_CKPT" ]; then
    run_stage "born_again_gen2" "experiments/configs/born_again_p50.yaml" "$GEN1_CKPT" || true
else
    echo "[born_again_gen2] SKIPPED (Gen 1 checkpoint missing at $GEN1_CKPT)" >> "$SUMMARY_FILE"
fi

# --- STAGE 5: Born Again Gen 3 ---
GEN2_CKPT="$CHECKPOINT_ROOT/born_again_gen2/distillation_at_best.pth"
[ ! -f "$GEN2_CKPT" ] && GEN2_CKPT="$CHECKPOINT_ROOT/born_again_gen2/distillation_best.pth"

if [ -f "$GEN2_CKPT" ]; then
    run_stage "born_again_gen3" "experiments/configs/born_again_p50.yaml" "$GEN2_CKPT" || true
else
    echo "[born_again_gen3] SKIPPED (Gen 2 checkpoint missing at $GEN2_CKPT)" >> "$SUMMARY_FILE"
fi

# --- FINAL VISUALIZATIONS ---
echo ""
echo ">>> GENERATING FINAL t-SNE PLOTS"

BASELINE_4495="experiments/checkpoints/slurm-train-eval-4495/baseline_ls005_24f_lightaug/baseline_best.pth"
TSNE_OUT_DIR="$ROOT_DIR/tsne_plots"
mkdir -p "$TSNE_OUT_DIR"

# Comparison: Teacher R50 vs Baseline 4495 vs Last Born Again
LAST_BA_CKPT=""
for gen in born_again_gen3 born_again_gen2 born_again_gen1; do
    for name in distillation_at_best.pth distillation_best.pth; do
        if [ -f "$CHECKPOINT_ROOT/$gen/$name" ]; then
            LAST_BA_CKPT="$CHECKPOINT_ROOT/$gen/$name"
            break 2
        fi
    done
done

if [ -n "$LAST_BA_CKPT" ]; then
    echo "Running t-SNE Comparison: Last Born Again ($LAST_BA_CKPT)"
    apptainer run --nv \
        --env TMPDIR="$TMPDIR" \
        /shared/sifs/latest.sif \
        python -u -m src.evaluation.tsne_visualizer \
            --config "experiments/configs/born_again_p50.yaml" \
            --teacher-ckpt "$TEACHER_R50" \
            --baseline-ckpt "$BASELINE_4495" \
            --student-ckpt "$LAST_BA_CKPT" \
            --output-dir "$TSNE_OUT_DIR" \
            --output-filename "tsne_born_again_final" \
            --num-classes 10 || true
fi

echo "Pipeline completed. Summary: $SUMMARY_FILE"
