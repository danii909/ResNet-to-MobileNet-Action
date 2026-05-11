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

def extract_from_summary(summary_path):
    metrics = {}
    if not summary_path.exists():
        return metrics
    try:
        with open(summary_path, 'r') as f:
            content = f.read()
    except:
        return metrics
    
    # Common patterns: key: value
    patterns = {
        'best_eval_acc': [r'best_eval_acc:\s*([\d\.]+)', r'best_acc:\s*([\d\.]+)'],
        'final_eval_acc': [r'final_eval_acc:\s*([\d\.]+)'],
        'final_eval_top5': [r'final_eval_top5:\s*([\d\.]+)', r'eval_acc_top5:\s*([\d\.]+)'],
        'final_train_acc': [r'final_train_acc:\s*([\d\.]+)'],
        'final_test_acc': [r'final_test_acc:\s*([\d\.]+)', r'top1:\s*([\d\.]+)'],
        'final_test_top5': [r'final_test_top5:\s*([\d\.]+)', r'top5:\s*([\d\.]+)']
    }
    
    for key, regexes in patterns.items():
        for reg in regexes:
            match = re.search(reg, content, re.IGNORECASE)
            if match:
                metrics[key] = match.group(1)
                break
    return metrics

def parse_pipeline_summary(pipeline_path):
    # Ritorna un dizionario: { stage_name: { metric_key: value } }
    summary = {}
    if not pipeline_path.exists():
        return summary
    
    try:
        with open(pipeline_path, 'r') as f:
            content = f.read()
    except:
        return summary

    # Divide il file in sezioni basate sui titoli degli stage: [teacher], [baseline], [distillation], ecc.
    sections = re.split(r'\n\[([^\]]+)\]\n', '\n' + content)
    # sections[0] è il preambolo
    # Poi coppie (nome_sezione, contenuto_sezione)
    for i in range(1, len(sections), 2):
        stage_name = sections[i].strip()
        stage_content = sections[i+1]
        
        # Estrae i risultati dalle sottosezioni [train results] e [eval results] all'interno dello stage
        metrics = extract_from_summary_content(stage_content)
        summary[stage_name] = metrics
    return summary

def extract_from_summary_content(content):
    metrics = {}
    patterns = {
        'best_eval_acc': [r'best_eval_acc:\s*([\d\.]+)', r'best_acc:\s*([\d\.]+)'],
        'final_eval_acc': [r'final_eval_acc:\s*([\d\.]+)'],
        'final_eval_top5': [r'final_eval_top5:\s*([\d\.]+)', r'eval_acc_top5:\s*([\d\.]+)'],
        'final_train_acc': [r'final_train_acc:\s*([\d\.]+)'],
        'final_test_acc': [r'final_test_acc:\s*([\d\.]+)', r'top1:\s*([\d\.]+)'],
        'final_test_top5': [r'final_test_top5:\s*([\d\.]+)', r'top5:\s*([\d\.]+)']
    }
    for key, regexes in patterns.items():
        for reg in regexes:
            match = re.search(reg, content, re.IGNORECASE)
            if match:
                metrics[key] = match.group(1)
                break
    return metrics

def parse_run(run_dir):
    run_dir = Path(run_dir)
    job_match = re.search(r'-(\d+)', run_dir.name)
    job_id = int(job_match.group(1)) if job_match else 0
    job_tag = run_dir.name
    
    # Cerca pipeline_summary nel job_dir
    pipeline_metrics = parse_pipeline_summary(run_dir / 'pipeline_summary.txt')

    records = []
    # Trova tutte le configurazioni (una per ogni stage/esperimento nel job)
    for config_path in run_dir.rglob('config_summary.json'):
        train_dir = config_path.parent
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except:
            continue
        
        # Estrazione Metriche
        metrics = {}
        
        # 1. Da file metric_* (legacy/fallback)
        for metric_file in train_dir.glob('metric_*'):
            name = metric_file.name
            val = name.split('_')[-1]
            if 'best_eval_acc' in name or 'best_acc' in name:
                metrics['best_eval_acc'] = val
            elif 'final_eval_acc' in name:
                metrics['final_eval_acc'] = val
            elif 'final_eval_top5' in name:
                metrics['final_eval_top5'] = val
            elif 'final_train_acc' in name:
                metrics['final_train_acc'] = val
            elif 'final_test_acc' in name:
                metrics['final_test_acc'] = val
            elif 'final_test_top5' in name:
                metrics['final_test_top5'] = val

        # 2. Da training_summary.txt e evaluation_summary.txt
        metrics.update(extract_from_summary(train_dir / 'training_summary.txt'))
        # Controlla anche la cartella sibling 'eval' (se esiste)
        eval_summary_path = train_dir.parent / 'eval' / 'evaluation_summary.txt'
        metrics.update(extract_from_summary(eval_summary_path))
        # Controlla anche se training_summary.txt è nella cartella parent (per alcune strutture)
        metrics.update(extract_from_summary(train_dir.parent / 'training_summary.txt'))

        # 3. Da pipeline_summary.txt (match per stage name)
        # Determiniamo il nome dello stage dal path
        stage_name = None
        for sname in pipeline_metrics.keys():
            if sname in str(train_dir).lower() or (config.get('mode') and sname == config.get('mode')):
                stage_name = sname
                break
        
        if stage_name and stage_name in pipeline_metrics:
            metrics.update(pipeline_metrics[stage_name])

        # Normalizzazione exp_name
        exp_name = config.get('mode', 'unknown')
        path_str = str(train_dir).lower()
        if 'teacher' in path_str: exp_name = 'teacher'
        elif 'baseline' in path_str: exp_name = 'baseline'
        elif 'distillation' in path_str and 'at' not in path_str: exp_name = 'distillation'
        elif 'at_sym' in path_str: exp_name = 'at_sym'

        # Determina l'architettura
        mtype = config.get('model', {}).get('type')
        if mtype == 'teacher':
            arch = "ResNet-50"
        elif mtype == 'assistant':
            arch = "ResNet-18 (TA)"
        elif mtype == 'student':
            arch = "MobileNet3D"
        else:
            arch = mtype if mtype else "unknown"

        flat_config = flatten_dict(config)
        
        record = {
            'job_id': job_id,
            'job_tag': job_tag,
            'exp_name': exp_name,
            'architecture': arch,
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

    # Definizione Colonne in ordine richiesto
    # Mappatura: (chiave interna, nome visualizzato)
    perf_mapping = [
        ('final_test_acc', 'Test Top-1'),
        ('final_test_top5', 'Test Top-5'),
        ('best_eval_acc', 'Val Top-1'),
        ('final_eval_top5', 'Val Top-5'),
        ('final_train_acc', 'Train Acc')
    ]
    perf_cols = [m[0] for m in perf_mapping]
    perf_headers = {m[0]: m[1] for m in perf_mapping}

    # Aggiungiamo la colonna 'split' e 'architecture'
    meta_cols = ['job_id', 'split', 'architecture', 'job_tag', 'exp_name']
    
    all_keys = set()
    for r in all_records:
        # Calcolo del tipo di split in base all'ID
        if r['job_id'] < 4201:
            r['split'] = "No Group-Aware"
        elif r['job_id'] < 4492:
            r['split'] = "Group-Aware"
        else:
            r['split'] = "Group-Aware FIX"
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
        if h in perf_headers:
            return f"**{perf_headers[h]}**"
        return h.replace('training.', '').replace('distillation.', '').replace('dataset.', '').replace('model.', '')

    header = "| " + " | ".join(clean_header(c) for c in final_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(final_cols)) + " |"
    
    rows = []
    last_job_id = None
    last_split = None
    for r in all_records:
        # Inserisce un'intestazione di sezione se lo split cambia
        if last_split is not None and r['split'] != last_split:
            section_marker = f"| **SECTION** | **{r['split'].upper()}** | " + " | ".join([" "] * (len(final_cols)-2)) + " |"
            rows.append(section_marker)
        elif last_job_id is not None and r['job_id'] != last_job_id:
            # Spacer marcato tra Job ID differenti
            spacer = "| " + " | ".join(["---"] * len(final_cols)) + " |"
            rows.append(spacer)
        
        row = "| " + " | ".join(str(r.get(c, '')) for c in final_cols) + " |"
        rows.append(row)
        last_job_id = r['job_id']
        last_split = r['split']
        
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
