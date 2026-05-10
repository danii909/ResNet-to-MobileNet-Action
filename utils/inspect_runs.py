import json
from pathlib import Path

runs_dir = Path(r'c:\Development\KD_Project\wandb\matched_slurm_runs')

for job in ['4391', '4409']:
    print(f'\n=== Job {job} ===')
    for run in runs_dir.glob('*'):
        if not run.is_dir(): continue
        
        meta_path = run / 'files' / 'wandb-metadata.json'
        if not meta_path.exists(): continue
        
        with open(meta_path) as f:
            meta = json.load(f)
            
        slurm_id = str(meta.get('slurm', {}).get('job_id', ''))
        if slurm_id == job:
            program = meta.get('program', '')
            args = ' '.join(meta.get('args', []))
            
            print(f'- {run.name}:')
            print(f'  Program: {program}')
            print(f'  Args: {args}')
