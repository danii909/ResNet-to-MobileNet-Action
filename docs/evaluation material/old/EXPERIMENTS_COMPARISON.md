# Experiments Comparison (Temporal Order by SLURM ID)

## Scope
Comparison across these experiment folders:
- `slurm-train-seq-3998`
- `slurm-train-eval-4041`
- `slurm-multiple-runs-4077`
- `slurm-strongaug-runs-4129`
- `slurm-recovery-24f-sweep-4137`
- `slurm-24f-refine3-a-4141`
- `slurm-24f-refine3-b-4142`
- `slurm-baseline-24f-phase2-4147`
- `slurm-baseline-24f-phase3-4162`

For the protocol-level split classification of these runs, see [docs/SPLIT_PROTOCOL_AUDIT.md](docs/SPLIT_PROTOCOL_AUDIT.md).

Rows are grouped by comparison pattern (`teacher -> baseline -> student`) when possible.
Missing fields are intentionally left empty.

Split protocol note:
- Most of the rows below are train/test-only historical runs.
- `exp_id=4168` and `exp_id=4184` belong to the non-group-aware internal-eval family.
- `exp_id=4201` is the corrected group-aware family and is listed in [docs/EXPERIMENTS_COMPARISON_EXTENDED.md](docs/EXPERIMENTS_COMPARISON_EXTENDED.md).

## Teacher Version Mapping
Teacher performance is not unique across all experiments, so student runs are mapped to teacher versions:

- Teacher v1: `exp_id=3998`, run `run-1`, best test acc `83.29`
- Teacher v2: `exp_id=4041`, run `teacher`, best test acc `88.21`

Student-teacher association:
- Students in `exp_id=3998` use Teacher v1 (same sequential job).
- Students in `exp_id=4041, 4077, 4129, 4137, 4141, 4142` use Teacher v2 (teacher checkpoint chain based on 4041).
- Baselines in `exp_id=4147, 4162` are student-only runs (no teacher checkpoint used during training).

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
| 4147 | slurm-baseline-24f-phase2-4147 | baseline_for_kd_t8_a07_24f_lightaug | baseline | student | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  |  |  | v2 (indirect) |  | 66.98 | 57 | 99.77 | 66.38 | 88.13 | 66.98 | 88.00 |
| 4142 | slurm-24f-refine3-b-4142 | kd_t8_a07_24f_lightaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 8 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 70.18 | 58 | 96.51 | 69.71 | 90.46 | 70.18 | 90.40 |
| 4147 | slurm-baseline-24f-phase2-4147 | baseline_for_kd_t10_a07_24f_lightaug | baseline | student | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  |  |  | v2 (indirect) |  | 66.98 | 57 | 99.77 | 66.38 | 88.13 | 66.98 | 88.00 |
| 4137 | slurm-recovery-24f-sweep-4137 | kd_t10_a07_24f_lightaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.47 | 57 | 97.55 | 69.05 | 90.25 | 69.47 | 90.33 |
| 4147 | slurm-baseline-24f-phase2-4147 | baseline_for_kd_t12_a07_24f_lightaug | baseline | student | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  |  |  | v2 (indirect) |  | 66.98 | 57 | 99.77 | 66.38 | 88.13 | 66.98 | 88.00 |
| 4142 | slurm-24f-refine3-b-4142 | kd_t12_a07_24f_lightaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 12 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.87 | 52 | 97.80 | 69.13 | 90.93 | 69.87 | 90.35 |
| 4162 | slurm-baseline-24f-phase3-4162 | baseline_for_kd_t10_a07_24f_lightaug_warmup8 | baseline | student | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  |  |  | v2 (indirect) |  | 66.98 | 57 | 99.77 | 66.38 | 88.13 | 66.98 | 88.00 |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_warmup8 | distillation | student | 24 | 16 | 0.0005 | 0.01 |  | 8 | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.71 | 58 | 97.40 | 69.15 | 90.77 | 69.71 | 90.80 |
| 4162 | slurm-baseline-24f-phase3-4162 | baseline_for_kd_t10_a07_24f_minaug | baseline | student | 24 | 16 | 0.0005 | 0.01 | 0.1 |  |  |  |  | v2 (indirect) |  | 64.95 | 55 | 99.83 | 64.87 | 86.36 | 64.95 | 86.12 |
| 4137 | slurm-recovery-24f-sweep-4137 | kd_t10_a07_24f_minaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 67.25 | 54 | 97.91 | 66.75 | 89.35 | 67.25 | 89.53 |
| 4162 | slurm-baseline-24f-phase3-4162 | baseline_for_kd_t10_a07_24f_lightaug_ls005 | baseline | student | 24 | 16 | 0.0005 | 0.01 | 0.05 |  |  |  |  | v2 (indirect) |  | 69.05 | 58 | 99.64 | 68.97 | 88.24 | 69.05 | 88.21 |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_ls005 | distillation | student | 24 | 16 | 0.0005 | 0.01 | 0.05 |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 68.75 | 60 | 97.56 | 68.75 | 90.46 | 68.75 | 90.46 |
| 4137 | slurm-recovery-24f-sweep-4137 | kd_t10_a07_24f_lightaug_lr6e4 | distillation | student | 24 | 16 | 0.0006 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.05 | 58 | 96.67 | 68.94 | 91.17 | 69.05 | 91.25 |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_seed43 | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 71.03 | 60 | 98.16 | 71.03 | 90.93 | 71.03 | 90.93 |
| 4142 | slurm-24f-refine3-b-4142 | kd_t10_a07_24f_lightaug_stride2 | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 58.02 | 51 | 91.45 | 56.54 | 82.32 | 58.02 | 83.00 |
| 4129 | slurm-strongaug-runs-4129 | kd_t10_a07_strongaug | distillation | student | 16 | 24 | 0.0006 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 52.05 | 56 | 90.45 | 51.94 | 80.17 | 52.05 | 80.12 |
| 4129 | slurm-strongaug-runs-4129 | kd_t10_a07_strongaug_24f | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 55.83 | 60 | 92.18 | 55.83 | 83.27 | 55.83 | 83.27 |
| 4129 | slurm-strongaug-runs-4129 | kdat_t10_a07_strongaug | distillation_at | student | 16 | 24 | 0.0006 | 0.01 |  |  | 10 | 0.7 | 0.1 | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 52.23 | 56 | 90.63 | 52.00 | 79.49 | 52.23 | 79.67 |

## Quick Notes
- **Best student Top-1 so far: `71.03` (`exp_id=4141`, `kd_t10_a07_24f_lightaug_seed43`).**
- **Best baseline Top-1 so far: `69.05` (`exp_id=4162`, `baseline_for_kd_t10_a07_24f_lightaug_ls005`).**
- Best student Top-5 so far: `91.25` (`exp_id=4137`, `kd_t10_a07_24f_lightaug_lr6e4`).
- `exp_id=3998` has no evaluation folder; eval columns remain empty by design.

## Baseline Counterpart Analysis (Completed: exp_id=4147, 4162)

You completed the first 6 planned baseline counterparts (24-frame, seed=42). Below is the direct KD vs baseline comparison using matched settings.

### Mapping 1:1 (Student KD -> Baseline)

- kd_t8_a07_24f_lightaug -> baseline_for_kd_t8_a07_24f_lightaug
- kd_t10_a07_24f_lightaug -> baseline_for_kd_t10_a07_24f_lightaug
- kd_t12_a07_24f_lightaug -> baseline_for_kd_t12_a07_24f_lightaug
- kd_t10_a07_24f_lightaug_warmup8 -> baseline_for_kd_t10_a07_24f_lightaug_warmup8
- kd_t10_a07_24f_minaug -> baseline_for_kd_t10_a07_24f_minaug
- kd_t10_a07_24f_lightaug_ls005 -> baseline_for_kd_t10_a07_24f_lightaug_ls005

| Pair | KD run (top1) | Baseline run (top1) | KD lift (top1) | Note |
|---|---:|---:|---:|---|
| T8 lightaug | 70.18 | 66.98 | **+3.20** | Strong positive KD gain |
| T10 lightaug (parent) | 69.47 | 66.98 | **+2.49** | Positive KD gain |
| T12 lightaug | 69.87 | 66.98 | **+2.89** | Positive KD gain |
| warmup8 lightaug | 69.71 | 66.98 | **+2.73** | Positive KD gain |
| minaug | 67.25 | 64.95 | **+2.30** | Positive KD gain |
| ls005 lightaug | 68.75 | 69.05 | **-0.30** | KD under baseline |

### Considerazioni

1. KD e realmente utile nel regime 24f: 5/6 coppie hanno lift positivo tra +2.30 e +3.20 top1.
2. Il miglior guadagno e su T8 (+3.20), confermando che T=8 e il migliore tra i test con seed=42.
3. Il caso ls005 e molto informativo: baseline 69.05 > KD 68.75. Questo indica che con label smoothing 0.05 la distillazione peggiora, quindi ls=0.05 non e una buona combinazione con KD in questo setup.
4. I tre baseline Phase 2 (T8/T10/T12) sono identici in tutto cio che conta per il baseline (seed, optimizer, lr, wd, augment, ecc.), quindi avere lo stesso risultato 66.98 e coerente.
5. I baseline hanno final_train_acc molto alta (~99.6-99.8), segnale di overfit gia presente anche senza KD; la validazione su split separato resta prioritaria.

## Missing Baseline Counterparts (Updated)

After exp_id 4147 and 4162, these distillation runs still do not have a matching baseline in this table:

- `exp_id=3998`, run `run-3`
- `exp_id=4041`, run `distillation`
- `exp_id=4077`, run `kd_t6_a06`
- `exp_id=4077`, run `kd_t10_a07`
- `exp_id=4077`, run `kd_t6_a06_lr5e4_wd002`
- `exp_id=4129`, run `kd_t10_a07_strongaug`
- `exp_id=4129`, run `kd_t10_a07_strongaug_24f`
- `exp_id=4129`, run `kdat_t10_a07_strongaug`
- `exp_id=4137`, run `kd_t10_a07_24f_lightaug_lr6e4`
- `exp_id=4141`, run `kd_t10_a07_24f_lightaug_seed43` (deferred seed study)

## Next Step

Best next block (3 runs ~= 10h) for fair comparison coverage:

1. `exp_id=4077`, run `kd_t10_a07`
2. `exp_id=4077`, run `kd_t6_a06`
3. `exp_id=4041`, run `distillation`

These three close the most relevant historical 16-frame comparisons with strong signal value.
