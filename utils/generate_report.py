import os
import json
import re
from pathlib import Path

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_run(run_dir):
    run_dir = Path(run_dir)
    job_match = re.search(r'-(\d+)$', run_dir.name)
    job_id = int(job_match.group(1)) if job_match else 0
    job_tag = run_dir.name

    records = []
    
    # Cerca tutti i file config_summary.json ricorsivamente
    for config_path in run_dir.rglob('config_summary.json'):
        train_dir = config_path.parent
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception:
            continue
            
        flat_config = flatten_dict(config)
        
        # Estrae le performance dai file metric_*
        metrics = {}
        for metric_file in train_dir.glob('metric_*'):
            name = metric_file.name
            parts = name.split('_')
            try:
                val = parts[-1]
                float(val) # Verifica che sia un numero
                metric_key = "_".join(parts[1:-1])
                metrics[metric_key] = val
            except (ValueError, IndexError):
                continue
        
        # Determina il nome dell'esperimento in base al path
        exp_name = config.get('mode', 'unknown')
        path_str = str(train_dir).lower()
        if 'teacher' in path_str:
            exp_name = 'teacher'
        elif 'baseline' in path_str:
            exp_name = 'baseline'
        elif 'distillation' in path_str:
            exp_name = 'distillation'
        elif 'at_sym' in path_str:
            exp_name = 'at_sym'

        record = {
            'job_id': job_id,
            'job_tag': job_tag,
            'exp_name': exp_name,
            **metrics,
            **flat_config
        }
        records.append(record)
        
    return records

def get_exp_priority(exp_name):
    """Definisce la gerarchia: teacher > baseline > distillation."""
    exp_name = exp_name.lower()
    if 'teacher' in exp_name:
        return 0
    if 'baseline' in exp_name:
        return 1
    if 'distillation' in exp_name or 'at_sym' in exp_name or 'kd' in exp_name:
        return 2
    return 3

def generate_report(results_dir):
    results_dir = Path(results_dir)
    all_records = []
    
    # Itera su tutte le cartelle slurm-*
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith('slurm-'):
            all_records.extend(parse_run(item))
            
    # Ordina per job_id e poi per gerarchia interna
    all_records.sort(key=lambda x: (x['job_id'], get_exp_priority(x['exp_name']), x['exp_name']))
    
    if not all_records:
        print("No records found.")
        return

    # Definizione Colonne
    perf_cols = ['best_eval_acc', 'final_eval_acc', 'final_eval_top5']
    meta_cols = ['job_id', 'job_tag', 'exp_name']
    
    all_keys = set()
    for r in all_records:
        all_keys.update(r.keys())
    
    priority_hparams = [
        'training.lr', 
        'training.weight_decay', 
        'training.batch_size', 
        'distillation.alpha', 
        'distillation.temperature',
        'distillation.at_beta',
        'distillation.at_beta_spatial',
        'distillation.at_beta_temporal',
        'dataset.num_frames'
    ]
    
    other_cols = sorted([k for k in all_keys if k not in perf_cols and k not in meta_cols])
    other_cols.sort(key=lambda x: (x not in priority_hparams, priority_hparams.index(x) if x in priority_hparams else x))

    final_cols = meta_cols + perf_cols + other_cols
    final_cols = [c for c in final_cols if any(r.get(c) is not None for r in all_records)]
    
    # Pulizia nomi header
    def clean_header(h):
        return h.replace('training.', '').replace('distillation.', '').replace('dataset.', '').replace('model.', '')

    header = "| " + " | ".join(clean_header(c) for c in final_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(final_cols)) + " |"
    
    rows = []
    last_job_id = None
    for r in all_records:
        # Inserisce uno spacer marcato tra Job ID differenti
        if last_job_id is not None and r['job_id'] != last_job_id:
            spacer = "| " + " | ".join(["---"] * len(final_cols)) + " |"
            rows.append(spacer)
        
        row = "| " + " | ".join(str(r.get(c, '')) for c in final_cols) + " |"
        rows.append(row)
        last_job_id = r['job_id']
        
    report = f"# All Runs Report\n\nTotal runs found: {len(all_records)}\n\n"
    report += header + "\n" + separator + "\n" + "\n".join(rows)
    
    output_path = results_dir / 'report.md'
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"Report generato con successo in: {output_path}")

if __name__ == "__main__":
    # Path della cartella contenente le run
    # Utilizziamo un path relativo o configurabile se necessario
    current_file_path = Path(__file__).resolve()
    # Se il file è in utils/, i risultati sono in ../results/all-runs
    results_path = current_file_path.parent.parent / 'results' / 'all-runs'
    
    if not results_path.exists():
        # Fallback al path assoluto se il relativo non funziona
        results_path = Path(r'i:\Development 2.0\KD_Project\results\all-runs')
        
    generate_report(results_path)
