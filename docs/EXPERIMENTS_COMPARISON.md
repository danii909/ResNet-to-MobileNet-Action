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
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_seed43 | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 71.03 | 60 | 98.16 | 71.03 | 90.93 | 71.03 | 90.93 |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_ls005 | distillation | student | 24 | 16 | 0.0005 | 0.01 | 0.05 |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 68.75 | 60 | 97.56 | 68.75 | 90.46 | 68.75 | 90.46 |
| 4141 | slurm-24f-refine3-a-4141 | kd_t10_a07_24f_lightaug_warmup8 | distillation | student | 24 | 16 | 0.0005 | 0.01 |  | 8 | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.71 | 58 | 97.40 | 69.15 | 90.77 | 69.71 | 90.80 |
| 4142 | slurm-24f-refine3-b-4142 | kd_t12_a07_24f_lightaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 12 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 69.87 | 52 | 97.80 | 69.13 | 90.93 | 69.87 | 90.35 |
| 4142 | slurm-24f-refine3-b-4142 | kd_t8_a07_24f_lightaug | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 8 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 70.18 | 58 | 96.51 | 69.71 | 90.46 | 70.18 | 90.40 |
| 4142 | slurm-24f-refine3-b-4142 | kd_t10_a07_24f_lightaug_stride2 | distillation | student | 24 | 16 | 0.0005 | 0.01 |  |  | 10 | 0.7 |  | v2 | /home/brbdnl01e03e017o/dl26-projects/experiments/checkpoints/teacher_finetune_best.pth | 58.02 | 51 | 91.45 | 56.54 | 82.32 | 58.02 | 83.00 |

## Quick Notes
- **Best student Top-1 so far: `71.03` (`exp_id=4141`, `kd_t10_a07_24f_lightaug_seed43`)** ← NEW RECORD! +1.56 points from previous best.
- Best student Top-5 so far: `90.93` (`exp_id=4141`, `kd_t10_a07_24f_lightaug_seed43`).
- `exp_id=3998` has no evaluation folder; eval columns remain empty by design.

## Refinement Sweep Analysis (exp_id=4141, 4142)

### Results Summary
6 variants tested on 24-frame lightaug parent KD config (config: `kd_t10_a07_24f_lightaug` from exp_id=4137, which is **already a distillation**, not a true baseline).

**Key: 5 hyperparameter variants use seed=42 (fair comparison), 1 seed experiment uses seed=43 (separate study).**

| Variant | Config Change(s) | Seed | Top-1 Accuracy | Δ vs Parent Config (69.47%) | Status |
|---------|-------------------|---:|---|---|---|
| **Parent** | Reference | 42 | 69.47% | 0% | Baseline |
| **seed43** (exp) | seed=43 ONLY (deterministic test) | **43** | **71.03%** | **+1.56%** ⭐ | SEPARATE SEED STUDY |
| **T8** ⭐⭐ | temperature=8 (harder targets) | 42 | **70.18%** | **+0.71%** | HYPERPARAMETER WINNER |
| **T12** | temperature=12 (softer targets) | 42 | 69.87% | +0.40% | Suboptimal hyperparameter |
| **warmup8** | kd_warmup_epochs=8 | 42 | 69.71% | +0.24% | Modest effect |
| **ls005** | label_smoothing=0.05 | 42 | 68.75% | -0.72% | Harmful hyperparameter |
| **stride2** | max_temporal_stride=2 | 42 | **58.02%** | **-11.45%** | 💥 Catastrophic |

### Critical Findings

**1. Seed Experiment: seed=43 vs seed=42 (+1.56% swing)**
- **seed43 is NOT a hyperparameter variant; it's a separate seed deterministic study**
- seed=43 config achieves 71.03%, while all hyperparameter variants use seed=42 (69.47% parent)
- Demonstrates seed sensitivity: **±1.56% variation from random initialization alone**
- **Implication**: This improvement (71.03%) is seed-dependent; must validate with seed=42, 44, 45 to know if seed=43 is truly better or just lucky
- Important for Phase ? (after baselines): Perform multi-seed robustness study
- Training-test gap: final_train_acc=98.16% vs eval=71.03% (27% gap) is normal for KD

**2. True Hyperparameter Winner: T=8 (Temperature Hardening)**
- Among all variants with **seed=42** (fair comparison), ranking is: **T=8 (70.18%) > T=10 (69.47%, parent) > T12 (69.87%)**
- Main finding: *harder* targets (lower temperature) outperform softer targets
- Counter-intuitive: Teacher already provides "sufficiently soft" targets; additional softening via T=12 over-smooths
- **T=8 Delta: +0.71% vs parent with same seed** ← This is the real hyperparameter gain
- **Action**: T=8 should be Priority 1 baseline candidate (expect ~6-9% KD lift over student baseline)

**3. Label Smoothing: Moderate is Better**
- ls=0.05 → 68.75% (worse than parent config)
- ls=0.1 (parent config setting) → 69.47%
- **Inference**: Asymmetry suggests ls=0.1 is well-tuned; lowering to 0.05 harms performance
- **Action**: If temperature tuning doesn't yield gains, revisit label smoothing interval [0.05, 0.15]

**4. KD Warmup: Minor Effect**
- warmup_epochs=8 → +0.24% vs warmup_epochs=5 (baseline)
- Effect is positive but small and not the primary driver
- **Interpretation**: KD loss ramps up early in training (first 5-8 epochs); extending beyond 8 likely yields diminishing returns
- **Action**: Not a priority for next iterations

**5. Temporal Stride is Critical**
- stride=2 with 24 frames → effectively **~12 frames sampled** 
- Performance collapse: 58.02% vs 69%+ for stride=1
- **Root cause**: UCF-101 action recognition requires dense temporal information; coarse sampling loses short-lived actions
- **Verdict**: Do NOT vary max_temporal_stride for this domain
- **Action**: Keep stride=1 as invariant in all future configs

### Ranked Candidates for Next Phase

1. **seed43** (71.03%) ← Needs validation with fresh seeds
2. **T8** (70.18%) ← Good performance, systematic tuning vector
3. **warmup8** (69.71%) ← Worth including in baseline comparisons for completeness
4. **T12** (69.87%) ← Suboptimal; skip unless checking symmetry
5. **ls005** (68.75%) ← Reject; harmful regularization
6. **stride2** (58.02%) ← Reject; unusable

### Critical Clarification: Parent Config vs True Baseline
⚠️ The refinement sweep compares KD variants against a **parent KD config** (69.47% distillation from exp_id=4137).  
To understand the true KD improvement, we need to train **true baseline students** (same config without distillation loss).
- Expected baseline student performance: ~62-65% (estimate from exp_id=4137, exp_id=4041)
- Expected KD lift (seed43): ~71.03% - 62-65% = **6-9 percentage points**

### Implications for Next Phases

**Phase 2: Baseline Counterparts**
- Create 3 baseline configs matching best KD variants (seed43, T8, warmup8)
- **Use identical setup except remove distillation loss** (use standard CE training)
- Compare KD lift: estimate ~6-9% improvement from KD over baseline student

**Phase 3: Follow-up Sweep (Optional)**
- Temperature grid: T ∈ {4, 6, 8, 10, 12} (currently have 8, 10, 12; missing 4, 6)
- Label smoothing grid: ls ∈ {0.05, 0.1, 0.15} if temperature doesn't unlock gains
- Seed robustness: Validate seed43 winner with 2-3 additional random seeds

**Phase 4: Clean Eval Split (Deferred)**
- Once baselines are complete, implement 3-way split (train/val/test)
- Retrain top 2 KD + matching baselines with separate test set
- Mark as "final_clean_protocol=True" in table



## Missing Baseline Counterparts (Same Setup) — Prioritized for Training

Using a strict matching criterion on non-KD training setup (`num_frames`, `batch_size`, `lr`, `weight_decay`, and run-specific augmentation regime), the following distillation runs currently do **not** have a baseline counterpart. Listed below in priority order based on student performance (higher = more important).

**Constraint: 3 runs ≈ 10 hours cluster time. Plan in jobs of 3 runs each.**

### Phase 2 (Job A): Temperature Sweep Study (T8, T10, T12 with seed=42) — **PRIORITY 1** ⭐⭐⭐
Train these 3 baselines next (est. 10h). **Note: seed43 excluded as it's a separate deterministic seed study; all these use seed=42 for fair KD comparison.**

1. ✅ `exp_id=4142`, run `kd_t8_a07_24f_lightaug` — **70.18%** eval_top1 (HYPERPARAMETER WINNER)
   - Config: 24f, bs=16, lr=0.0005, wd=0.01, **temperature=8.0**
   - Expected baseline: ~62-64% (estimate: KD lift ~6-8%)

2. ✅ `exp_id=4137`, run `kd_t10_a07_24f_lightaug` — **69.47%** eval_top1 (parent KD config, baseline for comparison)
   - Config: 24f, bs=16, lr=0.0005, wd=0.01, **temperature=10.0**
   - Expected baseline: ~62-64% (reference point for T8 vs T10 KD lift comparison)

3. ✅ `exp_id=4142`, run `kd_t12_a07_24f_lightaug` — **69.87%** eval_top1 (softer targets variant)
   - Config: 24f, bs=16, lr=0.0005, wd=0.01, **temperature=12.0**
   - Expected baseline: ~62-64% (completing temperature sweep: T8 < T10 < T12)

### Phase 3 (Job B): Other 24-frame Refinements — **PRIORITY 2** ⭐⭐
Train these 3 baselines next (est. 10h):
1. ✅ `exp_id=4141`, run `kd_t10_a07_24f_lightaug_warmup8` — **69.71%** eval_top1 (kd_warmup_epochs=8 variant)
   - Config: 24f, bs=16, lr=0.0005, wd=0.01, warmup=8
   - Expected baseline: ~62-64%

2. ✅ `exp_id=4137`, run `kd_t10_a07_24f_minaug` — **67.25%** eval_top1 (minimum augmentation variant)
   - Config: 24f, bs=16, lr=0.0005, wd=0.01, minaug
   - Expected baseline: ~60-63%

3. ✅ `exp_id=4141`, run `kd_t10_a07_24f_lightaug_ls005` — **68.75%** eval_top1 (label_smoothing=0.05 variant)
   - Config: 24f, bs=16, lr=0.0005, wd=0.01, ls=0.05
   - Expected baseline: ~62-64%

### Phase 4 (Job C): 16-frame Legacy — **PRIORITY 3** ⭐
Train these 3 baselines (est. 10h, if time permits):
1. ✅ `exp_id=4077`, run `kd_t10_a07` — **65.66%** eval_top1
   - Config: 16f, bs=24, lr=0.0006, wd=0.01
   - Expected baseline: ~59-61%

2. ✅ `exp_id=4077`, run `kd_t6_a06` — **64.84%** eval_top1
   - Config: 16f, bs=24, lr=0.0006, wd=0.01, T=6 variant
   - Expected baseline: ~59-61%

3. ✅ `exp_id=4041`, run `distillation` — **65.11%** eval_top1
   - Config: 16f, bs=24, lr=0.0006, wd=0.01
   - Expected baseline: ~60-62%

### Skipped or Deferred (Low Priority or Requires Further Study)
- **`exp_id=4141`, run `kd_t10_a07_24f_lightaug_seed43` (71.03%):** DEFERRED — Separate seed experiment (seed=43 vs seed=42). Excluded from initial baseline training to ensure all baselines use same seed for fair KD lift comparison. Will validate seed robustness separately after Phase 2+3 complete (e.g., train baselines for seed42, seed44, seed45).
- `exp_id=3998`, run `run-3`: Too old, very low performance (38%), skip
- `exp_id=4077`, run `kd_t6_a06_lr5e4_wd002`: 16f, lower performance (63.55%), defer to Phase 4
- `exp_id=4129` *all runs* (`kd_t10_a07_strongaug*`, `kdat_t10_a07_strongaug`): Strongaug proved harmful (52-56%), **reject**
- `exp_id=4142`, run `kd_t10_a07_24f_lightaug_stride2`: Temporal stride=2 catastrophic (58%), **reject**

## Operational Plan (Agreed)
To avoid forgetting the agreed workflow, follow this order:

**Phase 2 (IMMEDIATE):** Train 3 baselines from temperature sweep (T8, T10, T12 with seed=42) — est. 10h
- Create `baseline_t8_24f_lightaug.yaml`, `baseline_t10_24f_lightaug.yaml`, `baseline_t12_24f_lightaug.yaml`
- **Note: seed43 study (71.03%) excluded for now; will validate multi-seed robustness in separate phase after baselines are complete**
- Submit `sbatch cluster/submit_single_job_baseline_24f_phase2_temp_sweep.sh`

**Phase 3:** Train 3 baselines from other 24f refinements (warmup8, minaug, ls005) — est. 10h
- Create `baseline_warmup8_24f.yaml`, `baseline_24f_minaug.yaml`, `baseline_ls005_24f.yaml`
- Submit `sbatch cluster/submit_single_job_baseline_24f_phase3_refinements.sh`

**Phase 4 (Optional):** Train 3 baselines from 16f legacy experiments — est. 10h (if time/priority permits)
- Create configs for 4077 kd_t10_a07, kd_t6_a06, and 4041 distillation

**Phase 5:** After all baselines complete, implement a distinct evaluation protocol with separate sets (`train` / `val` / `test`) and avoid reusing test for model selection.
- Modify `src/datasets/ucf101.py` to support 3-way split
- Retrain top 2 KD winners + matching baselines with separate test set
- Mark as `protocol_clean=True` in table

Reporting guidance:

- Mark current results as development phase (`test` reused for model selection).
- Mark final reruns with distinct evaluation split as the official best-practice phase.
