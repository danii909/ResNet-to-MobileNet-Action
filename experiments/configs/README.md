# Experiment Configurations

YAML configuration files for all training and evaluation runs.

## Directory Structure

```
configs/
├── experiments/          # Training configs
│   ├── teacher.yaml              # Teacher fine-tuning (3D ResNet-50)
│   ├── baseline.yaml             # Student trained from scratch
│   ├── distillation.yaml         # Logit-based KD (temperature, alpha)
│   ├── attention_transfer.yaml   # KD + Attention Transfer
│   ├── born_again.yaml           # Born-Again Networks (BAN)
│   └── cross_frame.yaml          # Asymmetric temporal distillation (16f/24f)
└── valid/                # Evaluation-only configs
    └── ...
```

## Usage

All training scripts accept a YAML config and optional CLI overrides:

```bash
python -m src.training.train --config experiments/configs/experiments/distillation.yaml \
    --override distillation.temperature=20 distillation.alpha=0.7
```

See the main [README.md](../../README.md) for full training instructions.
