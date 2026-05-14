from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    print("Missing plotting dependencies:", e)
    print("Install pandas, matplotlib, seaborn in your environment and retry.")
    sys.exit(1)

sns.set(style="whitegrid")

# Navigate to project root (parent of scripts directory)
PROJECT_ROOT = Path(__file__).parent.parent
ROOT = PROJECT_ROOT / "results" / "Train-eval-test-split (group-aware)"
if not ROOT.exists():
    print(f"Root folder not found: {ROOT}")
    sys.exit(1)

metrics_files = list(ROOT.rglob("metrics_epoch.csv"))
if not metrics_files:
    print("No metrics_epoch.csv files found under", ROOT)
    sys.exit(0)

aggregate = []
kd_students = []  # List of (exp_name, best_eval, temperature, alpha)

for mf in metrics_files:
    try:
        df = pd.read_csv(mf)
    except Exception as e:
        print(f"Failed to read {mf}: {e}")
        continue

    # Experiment dir: two levels up (train -> <phase> -> experiment root may vary)
    # We'll pick parent.parent as experiment dir to store figures
    experiment_dir = mf.parents[1]
    exp_name = experiment_dir.name
    out_dir = experiment_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = df["epoch"] if "epoch" in df.columns else df.index

    # Loss plot
    plt.figure(figsize=(8, 4))
    if "train_loss" in df.columns:
        plt.plot(epochs, df["train_loss"], label="train_loss")
    if "eval_loss" in df.columns:
        plt.plot(epochs, df["eval_loss"], label="eval_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss vs Epoch - {exp_name}")
    plt.legend()
    plt.tight_layout()
    loss_path = out_dir / f"{exp_name}_loss.png"
    plt.savefig(loss_path)
    plt.close()

    # Accuracy plot
    plt.figure(figsize=(8, 4))
    if "train_acc" in df.columns:
        plt.plot(epochs, df["train_acc"], label="train_acc")
    if "eval_acc" in df.columns:
        plt.plot(epochs, df["eval_acc"], label="eval_acc")
    if "eval_acc_top5" in df.columns:
        plt.plot(epochs, df["eval_acc_top5"], label="eval_acc_top5")
    plt.xlabel("epoch")
    plt.ylabel("accuracy (%)")
    plt.title(f"Accuracy vs Epoch - {exp_name}")
    plt.legend()
    plt.tight_layout()
    acc_path = out_dir / f"{exp_name}_accuracy.png"
    plt.savefig(acc_path)
    plt.close()

    # LR plot (if present)
    if "lr" in df.columns:
        plt.figure(figsize=(8, 3))
        plt.plot(epochs, df["lr"]) 
        plt.xlabel("epoch")
        plt.ylabel("lr")
        plt.title(f"Learning Rate - {exp_name}")
        plt.tight_layout()
        lr_path = out_dir / f"{exp_name}_lr.png"
        plt.savefig(lr_path)
        plt.close()

    # Summary text file
    best_eval = None
    if "eval_acc" in df.columns:
        best_eval = float(df["eval_acc"].max())
    elif "best_eval_acc" in df.columns:
        best_eval = float(df["best_eval_acc"].max())

    summary_lines = []
    summary_lines.append(f"experiment: {exp_name}")
    if best_eval is not None:
        summary_lines.append(f"best_eval_acc: {best_eval:.4f}")
        aggregate.append((exp_name, best_eval))
        # Check for student distillation config to extract temperature
        config_path = experiment_dir / "train" / "config_summary.json"
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                if config_data.get("profile") == "student_distillation" and config_data.get("distillation"):
                    temp = config_data["distillation"].get("temperature")
                    alpha = config_data["distillation"].get("alpha")
                    if temp is not None:
                        kd_students.append((exp_name, best_eval, temp, alpha))
            except Exception:
                pass  # ignore config read errors
    if "epoch" in df.columns and "train_loss" in df.columns:
        summary_lines.append(f"epochs: {int(df['epoch'].max())+1}")
        summary_lines.append(f"final_train_loss: {df['train_loss'].iloc[-1]:.6f}")
    if "eval_loss" in df.columns:
        summary_lines.append(f"final_eval_loss: {df['eval_loss'].iloc[-1]:.6f}")

    (out_dir / f"{exp_name}_metrics_summary.txt").write_text("\n".join(summary_lines))
    print(f"Wrote plots for {exp_name} -> {out_dir}")

# Plot comparativo per distillazione: accuracy vs temperatura
if kd_students:
    kd_df = pd.DataFrame(kd_students, columns=["exp_name", "best_eval_acc", "temperature", "alpha"])
    print(f"\nFound {len(kd_df)} distillation experiments:")
    print(kd_df.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy vs Temperatura (colore per alpha)
    ax1 = axes[0]
    unique_alphas = sorted(kd_df["alpha"].dropna().unique())
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i / max(len(unique_alphas), 1)) for i in range(len(unique_alphas))]

    for i, alpha in enumerate(unique_alphas):
        subset = kd_df[kd_df["alpha"] == alpha]
        ax1.scatter(subset["temperature"], subset["best_eval_acc"],
                    label=f"α={alpha}", color=colors[i], s=100, edgecolors="black")

    ax1.set_xlabel("Temperature (T)", fontsize=12)
    ax1.set_ylabel("Best Eval Accuracy (%)", fontsize=12)
    ax1.set_title("Accuracy vs Temperature (colored by α)", fontsize=14)
    ax1.legend(title="Alpha")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Bar chart raggruppato per temperatura
    ax2 = axes[1]
    grouped = kd_df.groupby("temperature")["best_eval_acc"].agg(["mean", "std"]).reset_index()
    x = range(len(grouped))
    bars = ax2.bar(x, grouped["mean"], yerr=grouped["std"], capsize=5,
                   color="steelblue", edgecolor="black", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T={t}" for t in grouped["temperature"]])
    ax2.set_xlabel("Temperature", fontsize=12)
    ax2.set_ylabel("Mean Accuracy (%)", fontsize=12)
    ax2.set_title("Mean Accuracy per Temperature", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    kd_plot_path = ROOT / "distillation_comparison.png"
    plt.savefig(kd_plot_path)
    plt.close()
    print(f"\nSaved distillation comparison plot -> {kd_plot_path}")

    kd_csv_path = ROOT / "distillation_results.csv"
    kd_df.to_csv(kd_csv_path, index=False)
    print(f"Saved distillation data -> {kd_csv_path}")

if aggregate:
    agg_df = pd.DataFrame(aggregate, columns=["exp_name", "best_eval_acc"])
    agg_df = agg_df.sort_values("best_eval_acc", ascending=False)
    print("\n=== All Experiments Summary ===")
    print(agg_df.to_string(index=False))

print("Done.")
