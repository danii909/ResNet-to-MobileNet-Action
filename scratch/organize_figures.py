import os
from pathlib import Path
import shutil

# Path to the figures directory
figures_dir = Path(r"i:\Development 2.0\KD_Project\results\Train-eval-test-split (group-aware)\slurm-train-eval-4577\predictions\figures")

if not figures_dir.exists():
    print(f"Directory not found: {figures_dir}")
    exit(1)

# Recognized suffixes
suffixes = ["_confusion_matrix.png", "_full.png", "_norm.png", "_top25.png", "_interactive.html"]

files = list(figures_dir.glob("*"))
for f in files:
    if f.is_dir():
        continue
    
    # Identify which suffix this file has
    experiment_name = None
    for s in suffixes:
        if f.name.endswith(s):
            experiment_name = f.name[:-len(s)]
            # Clean up 'predictions_' prefix for cleaner folder names
            if experiment_name.startswith("predictions_"):
                experiment_name = experiment_name[len("predictions_"):]
            break
    
    if experiment_name:
        dest_dir = figures_dir / experiment_name
        dest_dir.mkdir(exist_ok=True)
        
        # Determine new filename (remove the experiment name prefix to keep it clean in the subfolder)
        # Actually, let's keep the filenames as is for now, but moving them is the main goal.
        shutil.move(str(f), str(dest_dir / f.name))
        print(f"Moved {f.name} -> {experiment_name}/")

print("Organization complete.")
