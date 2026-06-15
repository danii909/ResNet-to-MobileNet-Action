# Knowledge Distillation for Mobile Action Recognition

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-UCF--101-green)](https://www.crcv.ucf.edu/data/UCF101.php)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Report](https://img.shields.io/badge/Report-REPORT.md-blueviolet)](docs/REPORT.md)

> **Group G24 — Project 6**

A full Knowledge Distillation pipeline to compress a heavy 3D video recognition model into a mobile-ready student, with logit-based KD, Attention Transfer, Born-Again Networks, Cross-Frame Distillation, and AMP stability fixes.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Results](#-results)
  - [Logit-based KD — Temperature Ablation](#logit-based-kd--temperature-ablation-α--07)
  - [Attention Transfer](#attention-transfer-at)
  - [Born-Again Networks](#born-again-networks-ban--starting-from-t8)
  - [Cross-Frame Distillation](#cross-frame-distillation-asymmetric-temporal-sub-sampling)
- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Cluster Usage](#-cluster-usage)

---

## 🧠 Overview

Video action recognition is computationally expensive: state-of-the-art 3D CNNs and Video Transformers are incompatible with mobile and edge devices. This project addresses this with **Knowledge Distillation (KD)**: a powerful 3D ResNet-50 **Teacher** model transfers its knowledge to an ultra-lightweight MobileNet3D **Student**, achieving a significant accuracy recovery with a **13.5× compression factor**.

**Key techniques explored:**

| Method | Description |
|--------|-------------|
| **Logit-based KD** | Student mimics teacher's soft probability distributions (Hinton et al., 2015) |
| **Temperature Ablation** | Systematic sweep over $T \in \{1, 5, 8, 10, 20\}$ to maximize dark knowledge transfer |
| **Attention Transfer (AT)** | Feature-level distillation via spatial + temporal attention maps, with custom fixes for capacity-gap mismatch |
| **Born-Again Networks** | Iterative self-distillation across 3 generations |
| **Cross-Frame Distillation** | Asymmetric temporal input: Teacher sees 24 frames, Student sees 16 — trades −3.17% accuracy for **2.11× inference speedup** |
| **AMP Stability** | FP16 overflow prevention via explicit float32 casting in attention computation |

---

## 🏗 Architecture

### Teacher — 3D ResNet-50 (`slow_r50`)

- Backbone from [SlowFast (Feichtenhofer et al., ICCV 2019)](https://arxiv.org/abs/1812.03982)
- Pretrained on **Kinetics-400**, fine-tuned on **UCF-101**
- Standard 3D convolutions across 7 residual blocks
- **Size:** ~128 MB | **Test Top-1:** 88.82%

### Student — MobileNet3D

- Adaptation of [MobileNetV2 (Sandler et al., CVPR 2018)](https://arxiv.org/abs/1801.04381) to 3D video
- Depthwise separable 3D convolutions + inverted residual blocks with linear bottlenecks
- **Size:** ~9.5 MB (~13.5× smaller) | **Baseline Test Top-1:** 59.48%

### Input Format

```
Videos: UCF-101 | 101 classes | 24 frames per clip | 112×112 resolution
Split:  Group-aware (seed=42) — avoids data leakage across train/val/test
```

---

## 📊 Results

### Logit-based KD — Temperature Ablation ($\alpha = 0.7$)

| Temperature ($T$) | Best Val Top-1 | Best Epoch | Train Acc | **Test Top-1** | Test Top-5 | Δ vs Baseline |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| *Baseline (scratch)* | *59.18%* | — | — | *59.48%* | *82.77%* | — |
| $T = 1$ | 62.75% | 60 | 98.44% | 62.91% | 85.59% | +3.43% |
| $T = 5$ | 63.12% | 57 | 95.87% | 62.07% | 87.02% | +2.59% |
| $T = 8$ | 64.57% | 56 | 97.59% | 64.31% | 87.68% | +4.83% |
| $T = 10$ | 64.85% | 60 | 98.28% | 64.47% | 87.44% | +4.99% |
| **$T = 20$ ✓ Best** | **65.18%** | 59 | 99.05% | **65.85%** | **87.81%** | **+6.37%** |

> Higher temperatures reveal *dark knowledge* (inter-class similarity), enabling the student to better learn decision boundaries. $T=20$ is optimal, recovering **+6.37 pp** over the scratch baseline.

### Attention Transfer (AT)

| Configuration | Val Top-1 | Δ vs Best KD |
|:-:|:-:|:-:|
| KD ($T=20$) | 65.18% | — |
| AT Symmetric ($\beta_s=0.05, \beta_t=0.05$) | 58.25% | −6.93% |
| AT Temporal-Only ($\beta_s=0, \beta_t=0.10$) | 59.04% | −6.14% |

> AT causes regression due to the **capacity gap**: depthwise-separable convolutions in MobileNet3D cannot replicate the spatial attention maps of the teacher's full 3D convolutions. The temporal signal is more transferable than the spatial one.

### Born-Again Networks (BAN) — Starting from $T=8$

| Generation | Teacher Source | Val Top-1 | Test Top-1 | Δ vs Gen 0 |
|:-:|:-:|:-:|:-:|:-:|
| Gen 0 (KD $T=8$) | ResNet-50 | 64.57% | 64.31% | — |
| Gen 1 | Gen 0 Student | 60.78% | 60.98% | −3.33% |
| Gen 2 | Gen 1 Student | 61.20% | 60.43% | −3.88% |
| Gen 3 | Gen 2 Student | 59.98% | 59.21% | −5.10% |

> Generational collapse occurs because the initial student (Gen 0) has insufficient discriminative capacity to serve as an effective oracle. Errors are amplified across generations.

### Cross-Frame Distillation: Asymmetric Temporal Sub-Sampling

A **Pareto-optimal** technique for edge deployment: the Teacher processes the full **24-frame** clip while the Student is trained on a temporally sub-sampled **16-frame** version of the same clip (uniform spacing via `torch.linspace`), distilled with the best KD config ($T=20$, $\alpha=0.7$).

At inference time the Student only ever sees 16 frames, making it significantly faster without any architectural change.

| Metric | 24f Student (KD $T=20$) | **16f Student (Cross-Frame)** | Δ |
|:--|:-:|:-:|:-:|
| Test Top-1 Accuracy | 65.85% | **62.68%** | −3.17% |
| CPU Latency | baseline | **−52.6%** | **2.11× faster** |
| GPU Throughput | baseline | **+68.8%** | — |

> The 16-frame student achieves a **2.11× CPU speedup** (−52.6% latency) and **+68.8% GPU throughput** gain, at the cost of only 3.17 percentage points — a compelling trade-off for mobile and edge deployment.

---

## 📁 Project Structure

```
KD_Project/
├── src/
│   ├── models/
│   │   ├── teacher.py        # 3D ResNet-50 (slow_r50) wrapper with hook support
│   │   ├── student.py        # MobileNet3D with depthwise separable 3D convolutions
│   │   └── assistant.py      # Shared utilities (loss functions, AT computation)
│   ├── datasets/             # UCF-101 dataset loading with group-aware splits
│   ├── training/             # Training loop, KD loss, AT loss, AMP support
│   ├── evaluation/           # Test-set inference, Top-1/Top-5, latency benchmarks
│   └── utils/                # Logging, config parsing, t-SNE visualizations
├── experiments/
│   ├── configs/              # YAML configs for each experiment
│   │   ├── experiments/      # Main training configs (teacher, baseline, distillation, AT, BAN)
│   │   └── valid/            # Evaluation-specific configs
│   ├── checkpoints/          # Saved model weights (.pth)
│   ├── logs/                 # SLURM and training logs
│   └── Results/              # Experiment output summaries
├── cluster/
│   ├── train.sh              # SLURM training script
│   ├── eval.sh               # SLURM evaluation script
│   └── aliases.sh            # Bash aliases for cluster shortcuts
├── docs/
│   ├── REPORT.md             # Full technical report (IT)
│   └── GUIDA_TRAINING_EVAL.md  # Cluster usage guide (IT)
├── environment.yml           # Conda environment specification
└── requirements.txt          # Pip dependencies
```

---

## ⚙️ Setup

### 1. Clone & Create Environment

```bash
git clone https://github.com/danii909/KD_Project.git
cd KD_Project
conda env create -f environment.yml
conda activate dl-project
```

**Key dependencies:** PyTorch 2.x, pytorchvideo, decord, wandb, tensorboard, scikit-learn, seaborn.

### 2. Dataset

Download **UCF-101** from https://www.crcv.ucf.edu/data/UCF101.php.  
Extract videos and splits into:

```
data/
├── UCF-101/
│   ├── ApplyEyeMakeup/
│   │   ├── v_ApplyEyeMakeup_g01_c01.avi
│   │   └── ...
│   └── ...             # 101 action classes
└── ucfTrainTestlist/
    ├── classInd.txt
    ├── trainlist01.txt
    └── testlist01.txt
```

### 3. Teacher Pretrained Weights

Download `SLOW_8x8_R50.pyth` (Kinetics-400 pretrained slow_r50) and place it at:

```
experiments/checkpoints/SLOW_8x8_R50.pyth
```

Or set the environment variable `SLOW_R50_WEIGHTS=/path/to/SLOW_8x8_R50.pyth`.

---

## 🏋️ Training

All scripts accept YAML config files with optional CLI overrides via `--override key=value`.

### Step 1 — Fine-tune the Teacher (3D ResNet-50)

```bash
python -m src.training.train --config experiments/configs/experiments/teacher.yaml
```

### Step 2 — Train Baseline Student (from scratch)

```bash
python -m src.training.train --config experiments/configs/experiments/baseline.yaml
```

### Step 3 — Logit-based Knowledge Distillation

```bash
python -m src.training.train --config experiments/configs/experiments/distillation.yaml
```

Override temperature or alpha from CLI:

```bash
python -m src.training.train \
  --config experiments/configs/experiments/distillation.yaml \
  --override distillation.temperature=20 distillation.alpha=0.7
```

### Step 4 — Attention Transfer

```bash
python -m src.training.train --config experiments/configs/experiments/attention_transfer.yaml
```

### Step 5 — Cross-Frame Distillation (16f Student, 24f Teacher)

The Student receives a temporally sub-sampled clip (16 frames) while the Teacher processes the full 24-frame clip. Configure via `distillation.student_frames` in the YAML config:

```bash
python -m src.training.train \
  --config experiments/configs/experiments/distillation.yaml \
  --override distillation.temperature=20 distillation.alpha=0.7 distillation.student_frames=16
```

Or use a dedicated cross-frame config if available:

```bash
python -m src.training.train --config experiments/configs/experiments/cross_frame.yaml
```

---

## 📈 Evaluation

```bash
# Evaluate Teacher
python -m src.evaluation.evaluate \
  --config experiments/configs/experiments/teacher.yaml \
  --override evaluation.checkpoint=experiments/checkpoints/teacher_finetune_best.pth

# Evaluate best KD Student (T=20)
python -m src.evaluation.evaluate \
  --config experiments/configs/experiments/distillation.yaml \
  --override evaluation.checkpoint=experiments/checkpoints/distillation_best.pth

# Evaluate Baseline Student
python -m src.evaluation.evaluate \
  --config experiments/configs/experiments/baseline.yaml \
  --override evaluation.checkpoint=experiments/checkpoints/baseline_best.pth
```

---

## 🖥 Cluster Usage

The project includes full SLURM integration for HPC clusters.

### Load Aliases (recommended)

```bash
source cluster/aliases.sh
# Make persistent:
install-aliases
```

### Training on Cluster

```bash
# Single experiment
train baseline.yaml

# Sequential training (one SLURM job)
train-seq teacher.yaml baseline.yaml distillation.yaml

# Chained training (separate jobs with afterok dependency)
train-chain teacher.yaml baseline.yaml distillation.yaml

# Full pipeline: train + evaluate in one shot
train-and-eval
```

### Evaluation on Cluster

```bash
# Using sbatch directly
CONFIG=experiments/configs/experiments/teacher.yaml \
CHECKPOINT=experiments/checkpoints/teacher_finetune_best.pth \
sbatch cluster/eval.sh

# Using alias
evaluate experiments/configs/experiments/distillation.yaml experiments/checkpoints/distillation_best.pth
```

### Monitoring

```bash
myjobs                        # List active jobs
lastlog                       # Tail the most recent log
tail -f logs/slurm-train-<JOB_ID>.log
cat experiments/logs/slurm-eval-<JOB_ID>/evaluation_summary.txt
```

### Disk Management

```bash
clean          # Dry-run: show reclaimable space
clean --force  # Remove old SLURM logs and W&B offline runs
diskusage      # Breakdown by HF cache, checkpoints, W&B
```

---

## 📚 Key References

1. Hinton et al. (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531.
2. Zagoruyko & Komodakis (2017). *Paying More Attention to Attention*. ICLR.
3. Furlanello et al. (2018). *Born-Again Neural Networks*. ICML.
4. Feichtenhofer et al. (2019). *SlowFast Networks for Video Recognition*. ICCV.
5. Sandler et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.

> 📖 For full theoretical background, architecture details, bug analysis, and group contributions, see **[docs/REPORT.md](docs/REPORT.md)**.

