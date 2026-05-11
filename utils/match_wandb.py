import os
import json
import shutil
from pathlib import Path

wandb_dir = Path(r"c:\Development\KD_Project\wandb")
target_jobs = ["4201", "4384", "4391", "4409"]
output_dir = wandb_dir / "matched_slurm_runs"

output_dir.mkdir(exist_ok=True)

for run_dir in wandb_dir.glob("offline-run-*"):
    if not run_dir.is_dir():
        continue
        
    metadata_path = run_dir / "files" / "wandb-metadata.json"
    if not metadata_path.exists():
        continue
        
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            
        is_match = False
        slurm_info = metadata.get("slurm", {})
        job_id = str(slurm_info.get("job_id", ""))
        
        if job_id in target_jobs:
            is_match = True
            print(f"Match found! Run: {run_dir.name} -> Job: {job_id}")
            
        if is_match:
            dest = output_dir / run_dir.name
            if not dest.exists():
                shutil.move(str(run_dir), str(dest))
                print(f"Moved {run_dir.name} to {dest}")
                
    except Exception as e:
        print(f"Error processing {run_dir}: {e}")

print("Done matching and moving runs.")
