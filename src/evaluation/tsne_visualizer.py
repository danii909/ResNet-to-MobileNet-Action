"""t-SNE visualization comparing Teacher and Student latent spaces.

Extra Objective: Provide t-SNE visualizations of the latent space to illustrate 
the structural differences in how the teacher and student map actions.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import yaml

from src.datasets.ucf101 import get_dataloaders
from src.models.student import get_student
from src.models.teacher import get_teacher

def get_class_names(dataset) -> list[str]:
    """Extract class names mapping from the dataset object."""
    if hasattr(dataset, "dataset"):  # Handle torch.utils.data.Subset
        dataset = dataset.dataset
        
    if hasattr(dataset, "class_to_idx"):
        # Local video dataset
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        return [idx_to_class.get(i, f"Class_{i}") for i in range(len(idx_to_class))]
    elif hasattr(dataset, "hf_dataset"):
        # Hugging Face dataset
        try:
            return dataset.hf_dataset.features["label"].names
        except (AttributeError, KeyError):
            pass
            
    # Fallback
    return [f"Class {i}" for i in range(101)]

def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE Visualizer for Teacher vs Student Latent Space")
    parser.add_argument("--config", type=str, required=True, help="Path to the training/evaluation config YAML file.")
    parser.add_argument("--teacher-ckpt", type=str, required=True, help="Path to the teacher model checkpoint.")
    parser.add_argument("--student-ckpt", type=str, required=True, help="Path to the student model checkpoint.")
    parser.add_argument("--baseline-ckpt", type=str, default=None, help="Optional path to the baseline student model checkpoint.")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of distinct classes to visualize (7-10 recommended).")
    parser.add_argument("--output-dir", type=str, default="experiments/logs", help="Directory to save the resulting plot.")
    parser.add_argument("--output-filename", type=str, default=None, help="Custom filename for the output plot (without extension).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    
    # 1. Initialization and Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable AMP if device is cuda
    use_amp = (device.type == "cuda")
    
    # Load Dataloader
    print("\nLoading test dataloader...")
    if "training" not in config:
        config["training"] = {}
    config["training"]["batch_size"] = config["training"].get("batch_size", 16)
    
    dataloaders = get_dataloaders(config)
    test_loader = dataloaders["test"]
    
    # 2. Class Filtering
    class_names = get_class_names(test_loader.dataset)
    num_total_classes = len(class_names)
    
    selected_class_ids = np.random.choice(num_total_classes, size=args.num_classes, replace=False)
    selected_class_ids.sort()
    selected_class_names = [class_names[i] for i in selected_class_ids]
    
    print(f"\nSelected classes for t-SNE ({len(selected_class_ids)}):")
    for cid, cname in zip(selected_class_ids, selected_class_names):
        print(f"  - {cid}: {cname}")
        
# 3. Load Models
    print("\nLoading Teacher model...")
    num_classes = config.get("dataset", {}).get("num_classes", 101)
    teacher = get_teacher(num_classes=num_classes, pretrained=False, checkpoint_path=args.teacher_ckpt)
    teacher.to(device)
    teacher.eval()
    
    print("Loading Student model...")
    width_mult = config.get("model", {}).get("width_mult", 1.0)
    if "student" in config.get("model", {}):
        width_mult = config["model"]["student"].get("width_mult", width_mult)
        
    student = get_student(num_classes=num_classes, width_mult=width_mult, checkpoint_path=args.student_ckpt)
    student.to(device)
    student.eval()
    
    baseline = None
    if args.baseline_ckpt:
        print("Loading Baseline model...")
        baseline = get_student(num_classes=num_classes, width_mult=width_mult, checkpoint_path=args.baseline_ckpt)
        baseline.to(device)
        baseline.eval()

    # --- INIZIO FIX: Forward Hook per il Teacher ---
    teacher_features_buffer = []
    def teacher_hook(module, input, output):
        # Flatten the output of the pooling layer: [B, C, 1, 1, 1] -> [B, C]
        teacher_features_buffer.append(output.flatten(1).detach().cpu().float().numpy())
    
    # In PyTorchVideo slow_r50, il pooling finale è solitamente in blocks[5].pool
    # o accessibile iterando i moduli. Questo cerca l'ultimo AdaptiveAvgPool3d.
    pool_layer = None
    for module in teacher.modules():
        if isinstance(module, torch.nn.AdaptiveAvgPool3d):
            pool_layer = module
    
    if pool_layer is not None:
        hook_handle = pool_layer.register_forward_hook(teacher_hook)
    else:
        print("Attenzione: AdaptiveAvgPool3d non trovato nel teacher. L'estrazione potrebbe fallire.")
    # --- FINE FIX ---

    # 4. Feature Extraction
    print("\nExtracting embeddings from the test set...")
    student_embeddings = []
    baseline_embeddings = []
    labels_list = []
    
    selected_set = set(selected_class_ids.tolist())
    
    pbar = tqdm(test_loader, desc="Extracting", leave=True)
    for inputs, targets in pbar:
        mask = torch.tensor([t.item() in selected_set for t in targets])
        if not mask.any():
            continue
            
        filtered_inputs = inputs[mask].to(device, non_blocking=True)
        filtered_targets = targets[mask]
        
        with torch.amp.autocast("cuda", enabled=use_amp):
            # Il passaggio del forward riempirà automaticamente teacher_features_buffer grazie all'hook
            _ = teacher(filtered_inputs) 
            s_emb = student.get_embedding(filtered_inputs)
            if baseline is not None:
                b_emb = baseline.get_embedding(filtered_inputs)
            
        student_embeddings.append(s_emb.detach().cpu().float().numpy())
        if baseline is not None:
            baseline_embeddings.append(b_emb.detach().cpu().float().numpy())
        labels_list.append(filtered_targets.numpy())

    # Rimuovi l'hook alla fine per pulizia
    if pool_layer is not None:
        hook_handle.remove()
        
    if not labels_list:
        print("Error: No samples found for the selected classes. Exiting.")
        return
        
    teacher_embeddings = np.concatenate(teacher_features_buffer, axis=0)
    student_embeddings = np.concatenate(student_embeddings, axis=0)
    if baseline is not None:
        baseline_embeddings = np.concatenate(baseline_embeddings, axis=0)
    labels_list = np.concatenate(labels_list, axis=0)
    
    print(f"\nExtracted {len(labels_list)} samples.")
    print(f"Teacher embeddings shape: {teacher_embeddings.shape}")
    print(f"Student embeddings shape: {student_embeddings.shape}")
    if baseline is not None:
        print(f"Baseline embeddings shape: {baseline_embeddings.shape}")
    
    # 5. t-SNE Computation
    print("\nComputing t-SNE for Teacher...")
    tsne = TSNE(n_components=2, random_state=args.seed, init='pca', learning_rate='auto')
    t_proj = tsne.fit_transform(teacher_embeddings)
    
    print("Computing t-SNE for Student...")
    s_proj = tsne.fit_transform(student_embeddings)
    
    if baseline is not None:
        print("Computing t-SNE for Baseline...")
        b_proj = tsne.fit_transform(baseline_embeddings)
    
    # 6. Visualization
    print("\nGenerating side-by-side plots...")
    string_labels = [class_names[lbl] for lbl in labels_list]
    
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#f8f9fa", "grid.color": "#e9ecef"})
    palette = sns.color_palette("husl", n_colors=args.num_classes)
    
    num_cols = 3 if baseline is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(8 * num_cols, 7), sharex=True, sharey=True)
    
    # If 1x2, axes is a 1D array. If 1x3, still a 1D array.
    ax_teacher = axes[0]
    
    # Left subplot: Teacher
    sns.scatterplot(
        x=t_proj[:, 0], y=t_proj[:, 1],
        hue=string_labels,
        hue_order=selected_class_names,
        palette=palette,
        ax=ax_teacher,
        alpha=0.8,
        s=50,
        legend=False 
    )
    ax_teacher.set_title("Teacher Latent Space (ResNet-50)", fontsize=14, pad=10)
    ax_teacher.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax_teacher.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax_teacher.tick_params(labelsize=10)
    
    if baseline is not None:
        ax_baseline = axes[1]
        sns.scatterplot(
            x=b_proj[:, 0], y=b_proj[:, 1],
            hue=string_labels,
            hue_order=selected_class_names,
            palette=palette,
            ax=ax_baseline,
            alpha=0.8,
            s=50,
            legend=False 
        )
        ax_baseline.set_title("Baseline Latent Space (MobileNet3D)", fontsize=14, pad=10)
        ax_baseline.set_xlabel("t-SNE Dimension 1", fontsize=12)
        ax_baseline.tick_params(labelsize=10)
        
        ax_student = axes[2]
    else:
        ax_student = axes[1]
        
    # Right/Last subplot: Student
    scatter_student = sns.scatterplot(
        x=s_proj[:, 0], y=s_proj[:, 1],
        hue=string_labels,
        hue_order=selected_class_names,
        palette=palette,
        ax=ax_student,
        alpha=0.8,
        s=50,
        legend=True
    )
    ax_student.set_title("Student Latent Space (Distilled)", fontsize=14, pad=10)
    ax_student.set_xlabel("t-SNE Dimension 1", fontsize=12)
    if baseline is None:
        ax_student.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax_student.tick_params(labelsize=10)
    
    # Shared Legend outside
    handles, labels_leg = scatter_student.get_legend_handles_labels()
    ax_student.get_legend().remove()
    
    fig.legend(
        handles, labels_leg, 
        loc='center right', 
        bbox_to_anchor=(1.08 if baseline is not None else 1.12, 0.5),  
        title="UCF-101 Classes",
        title_fontsize=15,
        fontsize=13,
        frameon=True,
        shadow=True,
        borderpad=1.5,
        facecolor="white",
        edgecolor="#cccccc"
    )
    
    plt.tight_layout(rect=[0, 0, 0.98, 1])
    
    # 7. Saving
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.output_filename:
        filename = f"{args.output_filename}.png"
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tsne_comparison_teacher_vs_student_{timestamp}.png"
        
    output_path = output_dir / filename
    
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"\nSuccess! t-SNE visualization saved to: {output_path}")

if __name__ == "__main__":
    main()
