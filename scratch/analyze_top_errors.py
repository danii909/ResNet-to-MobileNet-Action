import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path("i:/Development 2.0/KD_Project")
PREDS_DIR = PROJECT_ROOT / "results" / "Train-eval-test-split (group-aware)" / "predictions"

BASELINE_CSV = PREDS_DIR / "predictions_baseline_ls005_24f_lightaug.csv"
KD_CSV = PREDS_DIR / "predictions_kd_t20_a07_24f_lightaug.csv"

def get_error_rates(csv_path):
    df = pd.read_csv(csv_path)
    # Identify label and prediction columns
    label_col = 'label_name' if 'label_name' in df.columns else ('label' if 'label' in df.columns else 'label_idx')
    pred_col = 'pred_name' if 'pred_name' in df.columns else ('prediction' if 'prediction' in df.columns else 'pred_idx')
    
    # Calculate accuracy per class
    classes = sorted(df[label_col].unique())
    cm = confusion_matrix(df[label_col], df[pred_col], labels=classes)
    row_sums = cm.sum(axis=1)
    
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype('float') / row_sums[:, np.newaxis]
    accuracies = np.diag(cm_norm) * 100
    
    df_err = pd.DataFrame({
        'Class': classes,
        'Accuracy': accuracies,
        'ErrorRate': 100.0 - accuracies,
        'TotalSamples': cm.sum(axis=1)
    })
    
    # Let's also find what the top confusion is for each class
    top_confusions = []
    for i, cls in enumerate(classes):
        row = cm[i].copy()
        row[i] = 0 # zero out correct prediction
        if row.sum() > 0:
            most_confused_idx = row.argmax()
            most_confused_class = classes[most_confused_idx]
            most_confused_count = row[most_confused_idx]
            pct = (most_confused_count / row_sums[i]) * 100
            top_confusions.append(f"{most_confused_class} ({pct:.1f}%)")
        else:
            top_confusions.append("None")
    df_err['MostConfusedWith'] = top_confusions
    
    return df_err

df_base = get_error_rates(BASELINE_CSV)
df_kd = get_error_rates(KD_CSV)

print("=== BASELINE TOP 10 ERRORS ===")
print(df_base.sort_values(by='ErrorRate', ascending=False).head(10).to_string(index=False))

print("\n=== KD T=20 TOP 10 ERRORS ===")
print(df_kd.sort_values(by='ErrorRate', ascending=False).head(10).to_string(index=False))

# Merge to see relationship
merged = pd.merge(df_base, df_kd, on='Class', suffixes=('_base', '_kd'))
merged['DeltaErrorRate'] = merged['ErrorRate_kd'] - merged['ErrorRate_base']
merged['DeltaAccuracy'] = merged['Accuracy_kd'] - merged['Accuracy_base']

print("\n=== COMPARISON OF BASELINE TOP 5 ERRORS IN KD ===")
top5_base_classes = df_base.sort_values(by='ErrorRate', ascending=False).head(5)['Class'].tolist()
print(merged[merged['Class'].isin(top5_base_classes)].sort_values(by='ErrorRate_base', ascending=False).to_string(index=False))

print("\n=== COMPARISON OF KD TOP 5 ERRORS IN BASELINE ===")
top5_kd_classes = df_kd.sort_values(by='ErrorRate', ascending=False).head(5)['Class'].tolist()
print(merged[merged['Class'].isin(top5_kd_classes)].sort_values(by='ErrorRate_kd', ascending=False).to_string(index=False))
