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

# Path to results
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results" / "Train-eval-test-split (group-aware)"
PREDS_DIR = RESULTS_ROOT / "predictions"
OUT_DIR = RESULTS_ROOT / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Files to compare (Baseline vs T=20)
BASELINE_CSV = PREDS_DIR / "predictions_baseline_ls005_24f_lightaug.csv"
KD_CSV = PREDS_DIR / "predictions_kd_t20_a07_24f_lightaug.csv" 

def _resolve_prediction_columns(df: pd.DataFrame) -> tuple[str, str]:
    candidates = [
        ("label_name", "pred_name"),
        ("label", "prediction"),
        ("label_idx", "pred_idx"),
    ]

    for label_col, pred_col in candidates:
        if label_col in df.columns and pred_col in df.columns:
            return label_col, pred_col

    raise KeyError(
        "Nessuna coppia di colonne valida trovata. Attese una tra: "
        "('label_name', 'pred_name'), ('label', 'prediction'), ('label_idx', 'pred_idx')."
    )


def get_class_accuracies(csv_path: Path):
    if not csv_path.exists():
        print(f"File non trovato: {csv_path}")
        return None, None
        
    df = pd.read_csv(csv_path)
    label_col, pred_col = _resolve_prediction_columns(df)
    classes = sorted(df[label_col].unique())
    cm = confusion_matrix(df[label_col], df[pred_col], labels=classes)
    
    # Accuratezza per classe: diagonale divisa per somma della riga
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    accuracies = np.diag(cm_norm) * 100 # In percentuale
    
    return classes, accuracies

def plot_delta_accuracy():
    classes_base, acc_base = get_class_accuracies(BASELINE_CSV)
    classes_kd, acc_kd = get_class_accuracies(KD_CSV)
    
    if classes_base is None or classes_kd is None:
        return
        
    assert classes_base == classes_kd, "I due file hanno set di classi differenti!"
    
    # Calcolo del Delta
    delta = acc_kd - acc_base
    
    # Creazione DataFrame per facilitare l'ordinamento
    df_delta = pd.DataFrame({
        'Class': classes_base,
        'Delta': delta
    })
    
    # Ordina per delta
    df_delta = df_delta.sort_values(by='Delta', ascending=False)
    
    # Seleziona le top 10 migliorate e le top 5 peggiorate
    top_improved = df_delta.head(10)
    top_degraded = df_delta.tail(5)
    
    # Unisci per il plot
    plot_df = pd.concat([top_improved, top_degraded])
    # Ordina nuovamente per avere un gradiente visivo bello nel grafico
    plot_df = plot_df.sort_values(by='Delta', ascending=True)
    
    # Colori: verde per miglioramenti, rosso per peggioramenti
    colors = ['#e74c3c' if val < 0 else '#2ecc71' for val in plot_df['Delta']]
    
    # Plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(plot_df['Class'], plot_df['Delta'], color=colors, edgecolor='black', alpha=0.8)
    
    # Aggiungi etichette sui bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 1 if width > 0 else width - 1
        ha = 'left' if width > 0 else 'right'
        plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:+.1f}%', 
                 va='center', ha=ha, fontsize=10, fontweight='bold')

    plt.axvline(0, color='black', linewidth=1.5, linestyle='--')
    plt.xlabel('Delta Accuracy (%) (KD T=20 - Baseline)', fontsize=12)
    plt.title('Knowledge Distillation Impact per Class (Top Improvements vs Regressions)', fontsize=14, pad=20)
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Adjust margins to fit labels
    xlim = plt.xlim()
    plt.xlim(xlim[0] - 5, xlim[1] + 5)
    
    plt.tight_layout()
    
    out_path = OUT_DIR / "delta_accuracy_kd_vs_baseline.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nDelta Accuracy plot saved -> {out_path}")
    plt.close()

if __name__ == "__main__":
    plot_delta_accuracy()