import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

BASE = Path(r"I:\Development 2.0\KD_Project\results\Train-eval-test-split (group-aware)")

models = {
    "KD T=1\nα=0.7": BASE / r"Training\slurm-train-eval-4565\kd_t1_a07_24f_lightaug\eval\evaluation_summary.json",
    "KD T=5\nα=0.7": BASE / r"Training\slurm-train-eval-4565\kd_t5_a07_24f_lightaug\eval\evaluation_summary.json",
    "KD T=8\nα=0.7": BASE / r"Training\slurm-train-eval-4495\kd_t8_a07_24f_lightaug\eval\evaluation_summary.json",
    #"KD T=8 LS0.1\nα=0.7": BASE / r"Training\slurm-train-eval-4495\kd_t8_a07_24f_ls0.1_lightaug\eval\evaluation_summary.json",
    "KD T=10\nα=0.7": BASE / r"Training\slurm-train-eval-4565\kd_t10_a07_24f_lightaug\eval\evaluation_summary.json",
    "KD T=20\nα=0.7": BASE / r"Training\slurm-train-eval-4565\kd_t20_a07_24f_lightaug\eval\evaluation_summary.json",
    "AT temporal\nT=8 α=0.7": BASE / r"Training\slurm-train-eval-4511\at_temporal_only\eval\evaluation_summary.json",
    "AT symmetric\nT=8 α=0.7": BASE / r"Training\slurm-train-eval-4511\at_symmetric\eval\evaluation_summary.json",
    "AT temporal\nT=20 α=0.7": BASE / r"Training\slurm-train-eval-4601\at_temporal_only\eval\evaluation_summary.json",
    "AT symmetric\nT=20 α=0.7": BASE / r"Training\slurm-train-eval-4601\at_symmetric\eval\evaluation_summary.json",
    "Born Again \n Gen 1 \n T=2.5 α=0.5": BASE / r"Training\slurm-born-again-4529\born_again_gen1\eval\evaluation_summary.json",
    "Born Again \n Gen 2 \n T=2.5 α=0.5": BASE / r"Training\slurm-born-again-4529\born_again_gen2\eval\evaluation_summary.json",
    "Born Again \n Gen 3 \n T=2.5 α=0.5": BASE / r"Training\slurm-born-again-4529\born_again_gen3\eval\evaluation_summary.json",
    #"Teacher\nbaseline": BASE / r"Training\slurm-train-eval-4495\teacher_24f_lightaug\eval\evaluation_summary.json",
    "Baseline": BASE / r"Training\slurm-train-eval-4495\baseline_ls005_24f_lightaug\eval\evaluation_summary.json",
}


# ----- TOP1 -----
names = []
top1s = []
for label, path in models.items():
    with open(path) as f:
        data = json.load(f)
    names.append(label)
    top1s.append(data["top1"])

order = np.argsort(top1s)
names = [names[i] for i in order]
top1s = [top1s[i] for i in order]

best_idx = np.argmax(top1s)
colors = ["#4A90D9"] * len(names)
colors[best_idx] = "#E67E22"
baseline_idx = names.index("Baseline")
colors[baseline_idx] = "#2ECC71"

fig, ax = plt.subplots(figsize=(18, 6))
bars = ax.bar(range(len(names)), top1s, color=colors, edgecolor="black", linewidth=0.8, width=0.65)

for i, (name, val) in enumerate(zip(names, top1s)):
    ax.text(i, val + 0.15, f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

#ax.axhline(y=top1s[best_idx], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Best: {top1s[best_idx]:.2f}%")

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=10, rotation=0)
for label in ax.get_xticklabels():
    if "Baseline" in label.get_text():
        label.set_fontweight("bold")

ax.set_ylabel("Test Top-1 Accuracy (%)", fontsize=11)
ax.set_title("MObileNet3D — Test Set Top-1 Accuracy", fontsize=13, fontweight="bold")
ax.set_ylim(min(top1s) - 2, max(top1s) + 2.5)
#ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.annotate('Teacher top-1: 88.82%', 
            xy=(1.0, 1.0), xycoords='axes fraction',
            xytext=(-10, -10), textcoords='offset points',
            size=14, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
ba_label = "Born Again \n Gen 1 \n T=2.5 α=0.5"
if ba_label in names:
    ba_idx = names.index(ba_label)
    ba_val = top1s[ba_idx]
    ax.annotate('Teacher: KD T=8 α=0.7',
                xy=(ba_idx - 0.2, ba_val - 0.1), xycoords='data',
                xytext=(-35, 40), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5, connectionstyle="arc3,rad=-0.1", shrinkB=12),
                fontsize=9.5, fontweight='semibold', color='black',
                ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.7))

plt.tight_layout()
out = BASE / "distillation_histogram_test_top1.png"
plt.savefig(out, dpi=200)
plt.close()
print(f"Saved: {out}")

# ----- TOP5 -----
names = []
top5s = []
for label, path in models.items():
    with open(path) as f:
        data = json.load(f)
    names.append(label)
    top5s.append(data["top5"])

order = np.argsort(top5s)
names = [names[i] for i in order]
top5s = [top5s[i] for i in order]

best_idx = np.argmax(top5s)
colors = ["#4A90D9"] * len(names)
colors[best_idx] = "#E67E22"
baseline_idx = names.index("Baseline")
colors[baseline_idx] = "#2ECC71"

fig, ax = plt.subplots(figsize=(18, 6))
bars = ax.bar(range(len(names)), top5s, color=colors, edgecolor="black", linewidth=0.8, width=0.65)

for i, (name, val) in enumerate(zip(names, top5s)):
    ax.text(i, val + 0.15, f"{val:.2f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

#ax.axhline(y=top5s[best_idx], color="red", linestyle="--", linewidth=1.5, alpha=0.8, label=f"Best: {top5s[best_idx]:.2f}%")

ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, fontsize=10, rotation=0)
for label in ax.get_xticklabels():
    if "Baseline" in label.get_text():
        label.set_fontweight("bold")

ax.set_ylabel("Test Top-5 Accuracy (%)", fontsize=11)
ax.set_title("MObileNet3D — Test Set Top-5 Accuracy", fontsize=13, fontweight="bold")
ax.set_ylim(min(top5s) - 2, max(top5s) + 2.5)
#ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.annotate('Teacher top-5: 98.18%', 
            xy=(1.0, 1.0), xycoords='axes fraction',
            xytext=(-10, -10), textcoords='offset points',
            size=14, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))

ba_label = "Born Again \n Gen 1 \n T=2.5 α=0.5"
if ba_label in names:
    ba_idx = names.index(ba_label)
    ba_val = top5s[ba_idx]
    ax.annotate('Teacher: KD T=8 α=0.7',
                xy=(ba_idx - 0.2, ba_val - 0.1), xycoords='data',
                xytext=(-35, 40), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color="blue", lw=1.5, connectionstyle="arc3,rad=-0.1", shrinkB=12),
                fontsize=9.5, fontweight='semibold', color='black',
                ha='right', va='center',
                bbox=dict(boxstyle='round,pad=0.2', fc='w', ec='none', alpha=0.7))

plt.tight_layout()
out = BASE / "distillation_histogram_test_top5.png"
plt.savefig(out, dpi=200)
plt.close()
print(f"Saved: {out}")