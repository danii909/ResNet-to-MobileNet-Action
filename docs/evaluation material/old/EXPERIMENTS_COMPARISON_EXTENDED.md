# Experiments Comparison and Methodology

## Scope
This document summarizes the main KD/teacher/baseline experiments in the repo, explains how they were conducted, and adds the newest runs.

For the full row-by-row historical table, see [docs/EXPERIMENTS_COMPARISON.md](docs/EXPERIMENTS_COMPARISON.md).
For the split-protocol classification, see [docs/SPLIT_PROTOCOL_AUDIT.md](docs/SPLIT_PROTOCOL_AUDIT.md).

## Methodology

### Common training setup
Most runs in this comparison use the UCF-101 HF dataset backend with a fixed seed (`42`) and an official test evaluation at the end of training.

### Historical runs (exp_id 3998 to 4162)
These runs were produced before the latest split refactor and are mostly train/test-only. In practice, the training-time best score in those older jobs was usually derived from the official test split, not from a protected internal validation split. For the exact family assignment of each folder, use [docs/SPLIT_PROTOCOL_AUDIT.md](docs/SPLIT_PROTOCOL_AUDIT.md).

Other shared characteristics across the historical table:
- teacher versions are mapped as described below
- student runs use the MobileNet3D student model
- KD runs differ from baselines by teacher checkpoint, temperature, alpha, and sometimes warmup
- official test metrics are the final comparison metric

### Teacher version mapping
- Teacher v1: exp_id 3998, run `run-1`
- Teacher v2: exp_id 4041 and later historical student runs
- Teacher v3: the newer 24f teacher trained for the recent single-job pipeline

### New runs (exp_id 4184 and 4201)
These runs introduce the modern split protocol. `exp_id=4184` still uses an internal eval split that was not group-aware, while `exp_id=4201` uses the corrected group-aware split by `video_id`.

For the fair baseline-vs-KD comparison in exp_id 4201, all non-KD hyperparameters are shared:
- `dataset.use_eval_split=true`
- `dataset.eval_ratio=0.2`
- `dataset.split_seed=42`
- `dataset.num_frames=24`
- `dataset.crop_size=112`
- `dataset.resize_short_side=128`
- `dataset.use_random_resized_crop=false`
- `dataset.color_jitter_strength=0.1`
- `dataset.random_erasing_prob=0.05`
- `dataset.max_temporal_stride=1`
- `training.batch_size=16` for student runs
- `training.optimizer=adamw`
- `training.lr=0.0005`
- `training.weight_decay=0.01`
- `training.scheduler=cosine`
- `training.grad_clip=1.0`
- `training.mixed_precision=true`
- `training.label_smoothing=0.05`

Only KD-specific fields change in the distillation run:
- `training.mode=distillation`
- `training.kd_warmup_epochs=5`
- `distillation.teacher_checkpoint`
- `distillation.temperature`
- `distillation.alpha`

## Historical comparison summary

| exp_id | job | run | mode | num_frames | batch_size | lr | weight_decay | label_smoothing | kd_temp | kd_alpha | best_acc_train | final_test_acc | eval_top1 | note |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4041 | slurm-train-eval-4041 | teacher | teacher_finetune | 16 | 12 | 0.0005 | 0.001 |  |  |  | 88.21 | 88.21 | 88.21 | Teacher v2 |
| 4041 | slurm-train-eval-4041 | baseline | baseline | 16 | 24 | 0.0008 | 0.01 |  |  |  | 62.54 | 62.15 | 62.54 | Student only |
| 4041 | slurm-train-eval-4041 | distillation | distillation | 16 | 24 | 0.0006 | 0.01 |  | 8 | 0.7 | 65.11 | 64.47 | 65.11 | KD vs Teacher v2 |
| 4077 | slurm-multiple-runs-4077 | baseline_wd002_ls01 | baseline | 16 | 24 | 0.0008 | 0.02 |  |  |  | 62.12 | 61.38 | 62.12 | Baseline variant |
| 4077 | slurm-multiple-runs-4077 | kd_t6_a06 | distillation | 16 | 24 | 0.0006 | 0.01 |  | 6 | 0.6 | 64.84 | 64.58 | 64.84 | KD gain |
| 4077 | slurm-multiple-runs-4077 | kd_t10_a07 | distillation | 16 | 24 | 0.0006 | 0.01 |  | 10 | 0.7 | 65.66 | 65.66 | 65.66 | KD gain |
| 4077 | slurm-multiple-runs-4077 | kd_t6_a06_lr5e4_wd002 | distillation | 16 | 24 | 0.0005 | 0.02 |  | 6 | 0.6 | 63.55 | 63.05 | 63.55 | KD gain |
| 4147 | slurm-baseline-24f-phase2-4147 | baseline_t8_24f_lightaug | baseline | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  | 66.98 | 66.38 | 66.98 | 24f baseline |
| 4147 | slurm-baseline-24f-phase2-4147 | baseline_t10_24f_lightaug | baseline | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  | 66.98 | 66.38 | 66.98 | 24f baseline |
| 4147 | slurm-baseline-24f-phase2-4147 | baseline_t12_24f_lightaug | baseline | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  | 66.98 | 66.38 | 66.98 | 24f baseline |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_warmup8 | distillation | 24 | 16 | 0.0005 | 0.01 | 0.1 | 10 | 0.7 | 69.71 | 69.15 | 69.71 | KD gain |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_ls005 | distillation | 24 | 16 | 0.0005 | 0.01 | 0.05 | 10 | 0.7 | 68.75 | 68.75 | 68.75 | KD with ls005 |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_seed43 | distillation | 24 | 16 | 0.0005 | 0.01 |  | 10 | 0.7 | 71.03 | 71.03 | 71.03 | Best student so far |
| 4142 | slurm-24f-refine3-b-4142 | kd_t8_a07_24f_lightaug | distillation | 24 | 16 | 0.0005 | 0.01 |  | 8 | 0.7 | 70.18 | 69.71 | 70.18 | Best T8 24f |
| 4142 | slurm-24f-refine3-b-4142 | kd_t12_a07_24f_lightaug | distillation | 24 | 16 | 0.0005 | 0.01 |  | 12 | 0.7 | 69.87 | 69.13 | 69.87 | Strong KD |
| 4142 | slurm-24f-refine3-b-4142 | kd_t10_a07_24f_lightaug_stride2 | distillation | 24 | 16 | 0.0005 | 0.01 |  | 10 | 0.7 | 58.02 | 56.54 | 58.02 | Stride2 hurts |
| 4162 | slurm-baseline-24f-phase3-4162 | baseline_24f_minaug | baseline | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  | 64.95 | 64.87 | 64.95 | Minaug baseline |
| 4162 | slurm-baseline-24f-phase3-4162 | baseline_warmup8_24f_lightaug | baseline | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  | 66.98 | 66.38 | 66.98 | Baseline counterpart |
| 4162 | slurm-baseline-24f-phase3-4162 | baseline_ls005_24f_lightaug | baseline | 24 | 16 | 0.0005 | 0.01 | 0.05 |  |  | 69.05 | 68.97 | 69.05 | Best baseline so far |

## New experiments summary

| exp_id | job | run | mode | teacher_checkpoint | key setup | best_acc_train | final_test_acc | eval_top1 | eval_top5 | note |
|---:|---|---|---|---|---|---:|---:|---:|---:|---|
| 4184 | slurm-base-kd-t8-v3-4184 | baseline_ls005_24f_lightaug | baseline |  | 24f, ls005, shared student overrides | 94.68 | 64.84 | 64.84 | 86.97 | Baseline under updated split |
| 4184 | slurm-base-kd-t8-v3-4184 | kd_t8_a07_24f_lightaug | distillation | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_v3_24f.pth | 24f, T=8, alpha=0.7, shared student overrides | 92.15 | 67.33 | 67.33 | 89.08 | KD gain over baseline |
| 4201 | slurm-train-eval-4201 | teacher | teacher_finetune | experiments/checkpoints/slurm-train-eval-4201/teacher/teacher_finetune_best.pth | 24f teacher, train first | 92.31 | 91.66 | 88.71 | 98.18 | Teacher v3 in same job |
| 4201 | slurm-train-eval-4201 | baseline | baseline |  | 24f, same non-KD overrides as KD | 59.18 | 58.90 | 59.48 | 82.77 | Fair baseline counterpart |
| 4201 | slurm-train-eval-4201 | distillation | distillation | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/slurm-train-eval-4201/teacher/teacher_finetune_best.pth | 24f, same non-KD overrides as baseline, T=8, alpha=0.7 | 63.78 | 63.07 | 65.27 | 87.73 | Fair KD comparison |

## Takeaways

- The historical table is useful for studying KD behavior, but internal eval from the older split was optimistic.
- The new runs 4184 and 4201 are more comparable because the baseline and KD runs share the same non-KD setup.
- In the fair 4201 comparison, KD improves the baseline on the official test set.
- In the updated 24f setting, KD remains beneficial in the latest fair comparison, while the teacher is still much stronger than the student.
