# Experiments Comparison (Temporal Order by SLURM ID)

## Scope
Comparison across these experiment folders:
- `slurm-train-seq-3998`
- `slurm-train-eval-4041`
- `slurm-multiple-runs-4077`
- `slurm-strongaug-runs-4129`
- `slurm-recovery-24f-sweep-4137`

Rows are sorted by `exp_id` (time order).
Missing fields are intentionally left empty.

## Teacher Version Mapping
Teacher performance is not unique across all experiments, so student runs are mapped to teacher versions:

- Teacher v1: `exp_id=3998`, run `run-1`, best test acc `83.29`
- Teacher v2: `exp_id=4041`, run `teacher`, best test acc `88.21`

Student-teacher association:
- Students in `exp_id=3998` use Teacher v1 (same sequential job).
- Students in `exp_id=4041, 4077, 4129, 4137` use Teacher v2 (teacher checkpoint chain based on 4041).

## Main Table

| exp_id | experiment | run | mode | model | num_frames | batch_size | lr | weight_decay | label_smoothing | kd_warmup_epochs | kd_temp | kd_alpha | at_beta | teacher_version | teacher_checkpoint | best_acc_train | best_epoch | final_train_acc | final_test_acc | final_test_top5 | eval_top1 | eval_top5 |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3998 | slurm-train-seq-3998 | run-1 | teacher_finetune | teacher | 16 | 12 | 0.0005 | 0.0005 |  |  |  |  |  | v1 |  | 83.29 | 38 | 89.25 | 81.13 | 96.38 |  |  |
| 3998 | slurm-train-seq-3998 | run-2 | baseline | student | 16 | 24 | 0.0008 | 0.001 |  |  |  |  |  | v1 (indirect) |  | 35.66 | 59 | 99.70 | 35.34 | 59.42 |  |  |
| 3998 | slurm-train-seq-3998 | run-3 | distillation | student | 16 | 24 | 0.0006 | 0.001 |  |  | 3.0 | 0.4 |  | v1 | experiments/checkpoints/teacher_finetune_best.pth | 38.86 | 51 | 99.16 | 38.25 | 63.97 |  |  |
| 4041 | slurm-train-eval-4041 | teacher | teacher_finetune | teacher | 16 | 12 | 0.0005 | 0.001 |  |  |  |  |  | v2 |  | 88.21 | 33 | 91.96 | 88.21 | 98.12 | 88.21 | 98.26 |
| 4041 | slurm-train-eval-4041 | baseline | baseline | student | 16 | 24 | 0.0008 | 0.01 |  |  |  |  |  | v2 (indirect) |  | 62.54 | 56 | 99.69 | 62.15 | 84.35 | 62.54 | 84.38 |
| 4041 | slurm-train-eval-4041 | distillation | distillation | student | 16 | 24 | 0.0006 | 0.01 |  |  | 8 | 0.7 |  | v2 | experiments/checkpoints/teacher_finetune_best.pth | 65.11 | 56 | 95.96 | 64.47 | 88.47 | 65.11 | 88.37 |
| 4077 | slurm-multiple-runs-4077 | baseline_wd002_ls01 | baseline | student | 16 | 24 | 0.0008 | 0.02 |  |  |  |  |  | v2 (indirect) |  | 62.12 | 45 | 99.72 | 61.38 | 83.74 | 62.12 | 84.01 |
| 4077 | slurm-multiple-runs-4077 | kd_t6_a06 | distillation | student | 16 | 24 | 0.0006 | 0.01 |  |  | 6 | 0.6 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 64.84 | 59 | 97.03 | 64.58 | 88.47 | 64.84 | 88.21 |
| 4077 | slurm-multiple-runs-4077 | kd_t10_a07 | distillation | student | 16 | 24 | 0.0006 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 65.66 | 59 | 97.11 | 65.66 | 88.13 | 65.66 | 87.76 |
| 4077 | slurm-multiple-runs-4077 | kd_t6_a06_lr5e4_wd002 | distillation | student | 16 | 24 | 0.0005 | 0.02 |  |  | 6 | 0.6 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 63.55 | 58 | 97.36 | 63.05 | 88.05 | 63.55 | 87.76 |
| 4129 | slurm-strongaug-runs-4129 | kd_t10_a07_strongaug | distillation | student | 16 | 24 | 0.0006 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 52.05 | 56 | 90.45 | 51.94 | 80.17 | 52.05 | 80.12 |
| 4129 | slurm-strongaug-runs-4129 | kd_t10_a07_strongaug_24f | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 55.83 | 60 | 92.18 | 55.83 | 83.27 | 55.83 | 83.27 |
| 4129 | slurm-strongaug-runs-4129 | kdat_t10_a07_strongaug | distillation_at | student | 16 | 24 | 0.0006 | 0.01 |  |  | 10 | 0.7 | 0.1 | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 52.23 | 56 | 90.63 | 52.00 | 79.49 | 52.23 | 79.67 |
| 4137 | slurm-recovery-24f-sweep-4137 | kd_t10_a07_24f_minaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 67.25 | 54 | 97.91 | 66.75 | 89.35 | 67.25 | 89.53 |
| 4137 | slurm-recovery-24f-sweep-4137 | kd_t10_a07_24f_lightaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.47 | 57 | 97.55 | 69.05 | 90.25 | 69.47 | 90.33 |
| 4137 | slurm-recovery-24f-sweep-4137 | kd_t10_a07_24f_lightaug_lr6e4 | distillation | student | 24 | 16 | 0.0006 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.05 | 58 | 96.67 | 68.94 | 91.17 | 69.05 | 91.25 |

## Quick Notes
- Best student Top-1 so far: `69.47` (`exp_id=4137`, `kd_t10_a07_24f_lightaug`).
- Best student Top-5 so far: `91.25` (`exp_id=4137`, `kd_t10_a07_24f_lightaug_lr6e4`).
- `exp_id=3998` has no evaluation folder; eval columns remain empty by design.

## Missing Baseline Counterparts (Same Setup)
Using a strict matching criterion on non-KD training setup (`num_frames`, `batch_size`, `lr`, `weight_decay`, and run-specific augmentation regime), the following distillation runs currently do **not** have a baseline counterpart with the same configuration in this table:

- `exp_id=3998`, run `run-3` (`distillation`)
- `exp_id=4041`, run `distillation`
- `exp_id=4077`, run `kd_t6_a06`
- `exp_id=4077`, run `kd_t10_a07`
- `exp_id=4077`, run `kd_t6_a06_lr5e4_wd002`
- `exp_id=4129`, run `kd_t10_a07_strongaug`
- `exp_id=4129`, run `kd_t10_a07_strongaug_24f`
- `exp_id=4129`, run `kdat_t10_a07_strongaug`
- `exp_id=4137`, run `kd_t10_a07_24f_minaug`
- `exp_id=4137`, run `kd_t10_a07_24f_lightaug`
- `exp_id=4137`, run `kd_t10_a07_24f_lightaug_lr6e4`
