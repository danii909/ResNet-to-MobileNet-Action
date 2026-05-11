import os
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def extract_from_summary(summary_path):
    metrics = {}
    if not summary_path.exists():
        return metrics
    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return metrics
    
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
                metrics[key] = float(match.group(1))
                break
    return metrics

def parse_pipeline_summary(pipeline_path):
    summary = {}
    if not pipeline_path.exists():
        return summary
    try:
        with open(pipeline_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return summary

    sections = re.split(r'\n\[([^\]]+)\]\n', '\n' + content)
    for i in range(1, len(sections), 2):
        stage_name = sections[i].strip()
        stage_content = sections[i+1]
        metrics = {}
        patterns = {
            'best_eval_acc': [r'best_eval_acc:\s*([\d\.]+)', r'best_acc:\s*([\d\.]+)'],
            'final_train_acc': [r'final_train_acc:\s*([\d\.]+)'],
            'top1': [r'top1:\s*([\d\.]+)']
        }
        for key, regexes in patterns.items():
            for reg in regexes:
                match = re.search(reg, stage_content, re.IGNORECASE)
                if match:
                    metrics[key] = float(match.group(1))
                    break
        summary[stage_name] = metrics
    return summary

def parse_run(run_dir):
    run_dir = Path(run_dir)
    job_match = re.search(r'-(\d+)', run_dir.name)
    job_id = int(job_match.group(1)) if job_match else 0
    
    pipeline_metrics = parse_pipeline_summary(run_dir / 'pipeline_summary.txt')
    records = []
    
    for config_path in run_dir.rglob('config_summary.json'):
        train_dir = config_path.parent
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except: continue
        
        metrics = {}
        # 1. metric_*
        for metric_file in train_dir.glob('metric_*'):
            name = metric_file.name
            try:
                val = float(name.split('_')[-1])
                if 'best_eval_acc' in name or 'best_acc' in name: metrics['best_eval_acc'] = val
                elif 'final_train_acc' in name: metrics['final_train_acc'] = val
                elif 'final_test_acc' in name or 'top1' in name: metrics['final_test_acc'] = val
            except: continue

        # 2. summaries
        metrics.update(extract_from_summary(train_dir / 'training_summary.txt'))
        metrics.update(extract_from_summary(train_dir.parent / 'eval' / 'evaluation_summary.txt'))
        metrics.update(extract_from_summary(train_dir.parent / 'training_summary.txt'))

        # 3. pipeline
        for sname, s_metrics in pipeline_metrics.items():
            if sname in str(train_dir).lower() or (config.get('mode') and sname == config.get('mode')):
                if 'best_eval_acc' in s_metrics: metrics['best_eval_acc'] = s_metrics['best_eval_acc']
                if 'final_train_acc' in s_metrics: metrics['final_train_acc'] = s_metrics['final_train_acc']
                if 'top1' in s_metrics: metrics['final_test_acc'] = s_metrics['top1']

        exp_name = config.get('mode', 'unknown')
        if 'teacher' in str(train_dir).lower(): exp_name = 'teacher'
        elif 'baseline' in str(train_dir).lower(): exp_name = 'baseline'
        elif 'distillation' in str(train_dir).lower(): exp_name = 'distillation'

        if job_id < 4201: section = "No Group-Aware"
        elif job_id < 4492: section = "Group-Aware"
        else: section = "Group-Aware FIX"

        train_acc = metrics.get('final_train_acc', 0)
        test_acc = metrics.get('final_test_acc', metrics.get('best_eval_acc', 0))

        if train_acc > 0 or test_acc > 0:
            records.append({
                'job_id': job_id,
                'exp_name': exp_name,
                'section': section,
                'Train Acc': train_acc,
                'Test Acc': test_acc,
                'label': f"{job_id}-{exp_name}"
            })
    return records

def generate_plot(results_dir):
    results_dir = Path(results_dir)
    data = []
    for item in results_dir.iterdir():
        if item.is_dir() and item.name.startswith('slurm-'):
            data.extend(parse_run(item))
    
    if not data: return
    df = pd.DataFrame(data).sort_values(['job_id', 'exp_name'])
    df_melted = df.melt(id_vars=['job_id', 'label', 'section'], 
                        value_vars=['Train Acc', 'Test Acc'], 
                        var_name='Metric', value_name='Accuracy')

    sections = ["No Group-Aware", "Group-Aware", "Group-Aware FIX"]
    sections = [s for s in sections if s in df['section'].unique()]
    
    fig, axes = plt.subplots(len(sections), 1, figsize=(18, 7 * len(sections)), sharey=True)
    if len(sections) == 1: axes = [axes]

    for ax, section in zip(axes, sections):
        section_df = df_melted[df_melted['section'] == section]
        sns.barplot(data=section_df, x='label', y='Accuracy', hue='Metric', ax=ax)
        ax.set_title(f"Section: {section}", fontsize=18, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Accuracy (%)", fontsize=14)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(title='Metric', loc='upper left')

    plt.tight_layout()
    output_path = Path(__file__).parent / 'accuracy_histogram.png'
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved at {output_path}")

if __name__ == "__main__":
    generate_plot(r'i:\Development 2.0\KD_Project\results\all-runs')
