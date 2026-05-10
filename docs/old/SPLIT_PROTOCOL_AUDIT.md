# Split Protocol Audit

This document classifies the experiments in `results/` into three split protocols:

1. Train/test only: the official test split was also used as the training-time validation signal.
2. Train/eval/test, but eval was not group-aware: an internal eval split existed, but it was created by label stratification only, so clips from the same source video could still leak across train and eval.
3. Train/eval/test with group-aware eval: the corrected setup, where train/eval are split by label and by video group (`video_id`), so no source video is shared between train and eval.

For metric-by-metric comparison tables, see [docs/EXPERIMENTS_COMPARISON.md](docs/EXPERIMENTS_COMPARISON.md) and [docs/EXPERIMENTS_COMPARISON_EXTENDED.md](docs/EXPERIMENTS_COMPARISON_EXTENDED.md).

## How the classification was determined

Evidence used:
- `config_summary.json`
- `training_summary.txt`
- `pipeline_summary.txt`
- the chat history and the code change that introduced group-aware splitting in `src/datasets/ucf101.py`

Important rule:
- If `config_summary.json` does not contain `dataset.use_eval_split=true`, the run belongs to the train/test-only family.
- If `dataset.use_eval_split=true` is present before the group-aware patch, the run belongs to the non-group-aware family.
- If `dataset.use_eval_split=true` is present after the group-aware patch, the run belongs to the corrected group-aware family.

## Protocol 1: Train/Test only

These runs did not use a separate internal validation split. The model was trained while the official test split served as the validation/evaluation target during training, and the final evaluation also used the official test split.

### Runs in this family
- `slurm-train-seq-3998`
- `slurm-train-eval-4041`
- `slurm-multiple-runs-4077`
- `slurm-strongaug-runs-4129`
- `slurm-recovery-24f-sweep-4137`
- `slurm-24f-refine3-a-4141`
- `slurm-24f-refine3-b-4142`
- `slurm-baseline-24f-phase2-4147`
- `slurm-baseline-24f-phase3-4162`

### Why these are here
- Their `config_summary.json` files do not show `dataset.use_eval_split=true`.
- Their `training_summary.txt` files expose `final_test_acc` / `final_test_top5` rather than an internal eval metric.
- The `eval/` folders in the results are final official test inference, not evidence of an internal validation split.

### Practical consequence
- The reported training-time best metric is optimistic because it is tied to the test split.
- These runs are useful for historical comparison, but not for fair model selection.

## Protocol 2: Train/Eval/Test, but eval not group-aware

These runs introduced an internal eval split, but the split was initially label-only. That means the same source video could appear in both train and eval via different clips, which makes eval too easy.

### Runs in this family
- `TEACHER/slurm-train-4168`
- `slurm-base-kd-t8-v3-4184`

### Why these are here
- Their `config_summary.json` files explicitly contain `dataset.use_eval_split=true`.
- They were run before the group-aware split patch in `src/datasets/ucf101.py`.
- In the chat, these runs were the ones that produced suspiciously high internal eval values compared with the official test.

### Practical consequence
- Internal eval is better than pure train/test-only, but still too optimistic.
- It can still overestimate the true generalization gap because clips from the same video are not blocked from crossing train and eval.

## Protocol 3: Train/Eval/Test with group-aware eval

This is the corrected setup. Train and eval are split by label and by `video_id`, so a video contributes clips to only one side of the split.

### Runs in this family
- `slurm-train-eval-4201`

### Why this is here
- `config_summary.json` contains `dataset.use_eval_split=true`.
- The run happened after the group-aware split implementation in `src/datasets/ucf101.py`.
- The chat confirmed the new split was intended to be the permanent fix for the optimistic eval issue.

### Practical consequence
- The internal eval is much more trustworthy.
- The gap between internal eval and official test becomes interpretable instead of obviously inflated by leakage.

## Summary table

| Protocol | What is used during training | Leakage risk | Runs | Suitability for model selection |
|---|---|---:|---|---|
| Train/Test only | Official test split is also used as validation | High | 3998, 4041, 4077, 4129, 4137, 4141, 4142, 4147, 4162 | Low |
| Train/Eval/Test, non-group-aware | Internal eval split by label only | Medium-high | TEACHER/4168, 4184 | Medium-low |
| Train/Eval/Test, group-aware | Internal eval split by label + video group | Low | 4201 | High |

## Key interpretation

If you compare old runs against new runs, do not compare the old training-time best metric with the new official test metric. For a fair comparison:
- compare test to test, or
- compare internal eval to internal eval within the same split protocol.

The main conceptual jump in this repo is from:
- test-as-validation

to:
- internal eval without group protection

to:
- internal eval with group protection

That third step is the one that makes early stopping, checkpoint selection, and KD-vs-baseline comparisons genuinely meaningful.
