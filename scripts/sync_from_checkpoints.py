#!/usr/bin/env python3
import subprocess
from pathlib import Path
import sys

repo = Path.cwd()
wandb_dir = repo / 'wandb'
checkpoints_root = repo / 'experiments' / 'checkpoints'

jobs = ['slurm-train-eval-4201','slurm-train-eval-4384','slurm-at-temporal-4391','slurm-train-eval-4409']

found_runs = set()

for job in jobs:
    job_path = checkpoints_root / job
    if not job_path.exists():
        continue
    for p in job_path.rglob('*.pth'):
        rel = p.relative_to(repo).as_posix()  # experiments/checkpoints/...
        remote = f"/home/brbdnl01e03e017o/dl26-projects/{rel}"
        # search in wandb offline-run debug logs
        if wandb_dir.exists():
            for d in wandb_dir.iterdir():
                if not d.is_dir() or not d.name.startswith('offline-run-'):
                    continue
                log = d / 'logs' / 'debug.log'
                if log.exists():
                    try:
                        text = log.read_text(errors='ignore')
                    except Exception:
                        continue
                    if remote in text:
                        found_runs.add(str(d))

print('Found runs:')
for r in sorted(found_runs):
    print(r)

if not found_runs:
    print('No runs found matching checkpoints.')
    sys.exit(0)

# sync each run to desired project
project = 'kd-action-recognition-v2'
python_exe = sys.executable
for r in sorted(found_runs):
    print(f"Syncing {r} to project {project}...")
    cmd = [python_exe, '-m', 'wandb', 'sync', '--project', project, r]
    res = subprocess.run(cmd)
    print(f"Exit: {res.returncode}")

print('Done.')
