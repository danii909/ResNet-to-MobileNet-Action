from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
except Exception as e:
    print("Missing dependencies:", e)
    print("Install pandas, matplotlib, seaborn, scikit-learn and retry.")
    sys.exit(1)

# Check for plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Plotly not found. Interactive HTML plots will be skipped.")
    print("To enable interactive plots, run: pip install plotly")

RESULTS_ROOT = Path(__file__).parent.parent / "results"
if not RESULTS_ROOT.exists():
    print(f"Results folder not found: {RESULTS_ROOT}")
    sys.exit(1)

prediction_files = list(RESULTS_ROOT.rglob("predictions/*.csv"))
if not prediction_files:
    print("No prediction CSV files found under", RESULTS_ROOT)
    sys.exit(0)

def plot_and_save_cm(cm, classes, title, out_path, annot=True, figsize=(16, 14)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt=".2f" if cm.dtype == float else "d", 
                cmap="Blues", xticklabels=classes, yticklabels=classes,
                cbar_kws={"shrink": 0.75}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved -> {out_path}")

def plot_interactive_plotly(cm, classes, title, out_path):
    if not HAS_PLOTLY:
        return
    
    fig = px.imshow(cm,
                    labels=dict(x="Predicted", y="True", color="Count"),
                    x=classes,
                    y=classes,
                    title=title,
                    color_continuous_scale='Blues',
                    aspect="auto")
    
    fig.update_layout(
        width=1000,
        height=1000,
        xaxis_title="Predicted",
        yaxis_title="True"
    )
    
    fig.write_html(out_path)
    print(f"Saved Interactive -> {out_path}")

for pf in sorted(prediction_files):
    try:
        df = pd.read_csv(pf)
    except Exception as e:
        print(f"Failed to read {pf}: {e}")
        continue

    label_col = "label_name"
    pred_col = "pred_name"

    if label_col not in df.columns or pred_col not in df.columns:
        print(f"Skipping {pf}: missing required columns")
        continue

    # Build label ordering from sorted unique label names
    classes = sorted(df[label_col].unique())
    num_classes = len(classes)

    cm = confusion_matrix(df[label_col], df[pred_col], labels=classes)
    
    # Determine experiment name for subdirectory (remove 'predictions_' prefix)
    exp_name = pf.stem
    if exp_name.startswith("predictions_"):
        exp_name = exp_name[len("predictions_"):]
    
    out_dir = pf.parent / "figures" / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full Static Matrix (No annotations if too many classes)
    annot = num_classes <= 30
    plot_and_save_cm(cm, classes, f"Full CM - {pf.stem}", 
                     out_dir / f"{pf.stem}_full.png", annot=annot)

    # 2. Normalized Matrix (Percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    plot_and_save_cm(cm_norm, classes, f"Normalized CM - {pf.stem}", 
                     out_dir / f"{pf.stem}_norm.png", annot=False)

    # 3. Top-N Most Confused Classes
    # We identify classes with the lowest accuracy
    accuracies = np.diag(cm_norm)
    top_n = 25
    if num_classes > top_n:
        # Get indices of top_n lowest accuracies
        worst_indices = np.argsort(accuracies)[:top_n]
        worst_classes = [classes[i] for i in worst_indices]
        
        # Sub-matrix for these classes
        mask = df[label_col].isin(worst_classes)
        df_filtered = df[mask]
        
        cm_top = confusion_matrix(df_filtered[label_col], df_filtered[pred_col], labels=worst_classes)
        
        plot_and_save_cm(cm_top, worst_classes, f"Top {top_n} Worst Classes CM - {pf.stem}", 
                         out_dir / f"{pf.stem}_top{top_n}.png", annot=True, figsize=(12, 10))

    # 4. Interactive Plotly (if available)
    if HAS_PLOTLY:
        plot_interactive_plotly(cm, classes, f"Interactive CM - {pf.stem}", 
                               out_dir / f"{pf.stem}_interactive.html")

print("Done.")
