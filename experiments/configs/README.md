# Experiment Configurations

YAML configuration files for all training and evaluation runs.

## Directory Structure

```
configs/
├── main/                 # Main configuration files actually used for final results
│   ├── teacher_24f_evalsplit.yaml           # Main Teacher fine-tuning (3D ResNet-50)
│   ├── baseline_ls005_24f_lightaug.yaml    # Main Student baseline trained from scratch
│   ├── distillation_t8_a07_24f_lightaug.yaml # Main Logit-based KD (temperature=8, alpha=0.7)
│   ├── at_symmetric_t20.yaml                # Main Attention Transfer (Symmetric)
│   ├── at_temporal_t20.yaml                 # Main Attention Transfer (Temporal)
│   ├── born_again_p50.yaml                  # Main Born-Again Network (BAN)
│   └── crossframe_kd_t24_s16.yaml           # Main Cross-Frame Asymmetric Distillation (24f to 16f)
└── drafts/               # Drafts, tests, and failed hyperparameter trials
    └── ...
```

## Usage

All training scripts accept a YAML config and optional CLI overrides:

```bash
python -m src.training.train --config experiments/configs/main/distillation_t8_a07_24f_lightaug.yaml \
    --override distillation.temperature=20 distillation.alpha=0.7
```

See the main [README.md](../../README.md) for full training instructions.
