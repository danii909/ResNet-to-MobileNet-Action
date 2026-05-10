"""t-SNE visualization of teacher, baseline, and distilled student embedding spaces.

Extra Objective — Track 6: Knowledge Distillation for Mobile Action Recognition.

Usage (on cluster):
    python -m src.evaluation.tsne_visualizer \
        --config experiments/configs/distillation_t8_a07_24f_v2.yaml \
        --override \
            evaluation.teacher_checkpoint=/path/to/teacher_best.pth \
            evaluation.baseline_checkpoint=/path/to/baseline_best.pth \
            evaluation.distilled_checkpoint=/path/to/distillation_best.pth
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.ucf101 import get_dataloaders
from src.models.student import get_student
from src.models.teacher import get_teacher
from src.utils.config import get_config


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool = True,
    name: str = "model",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pre-classifier embeddings from model on the given dataloader.

    Args:
        model: Model with ``get_embedding(x)`` method.
        dataloader: DataLoader yielding (clips, labels).
        device: Target device.
        use_amp: Whether to use AMP for inference.
        name: Label used in the progress bar.

    Returns:
        Tuple of (embeddings [N, D], labels [N]).
    """
    model.eval()
    all_embs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    pbar = tqdm(dataloader, desc=f"Extracting [{name}]", leave=True, file=sys.stdout)
    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            emb = model.get_embedding(clips)  # [B, D]
        all_embs.append(emb.cpu().float().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_embs, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# t-SNE computation
# ---------------------------------------------------------------------------

def run_tsne(
    embeddings: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Run t-SNE reduction on the given embeddings.

    Args:
        embeddings: Input array [N, D].
        n_components: Output dimensionality.
        perplexity: t-SNE perplexity.
        n_iter: Number of optimization iterations.
        seed: Random seed for reproducibility.

    Returns:
        Reduced array [N, n_components].
    """
    from sklearn.manifold import TSNE

    n = embeddings.shape[0]
    print(
        f"  Running t-SNE: n={n}, D={embeddings.shape[1]}, "
        f"perplexity={perplexity}, n_iter={n_iter} ..."
    )
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=seed,
        init="pca",
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_tsne(
    tsne_xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    save_path: Path,
    num_classes: int = 101,
    max_classes_shown: int = 20,
) -> None:
    """Save a dark-themed t-SNE scatter plot.

    Args:
        tsne_xy: 2D coordinates [N, 2].
        labels: Integer class labels [N].
        title: Plot title.
        save_path: Destination PNG path.
        num_classes: Total number of classes in the dataset.
        max_classes_shown: Maximum number of classes to colour distinctly.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    unique_cls = np.unique(labels)

    # Select a visually distinct subset of classes if there are too many.
    rng = np.random.default_rng(42)
    if len(unique_cls) > max_classes_shown:
        shown_cls = rng.choice(unique_cls, max_classes_shown, replace=False)
        shown_cls = np.sort(shown_cls)
    else:
        shown_cls = unique_cls

    cmap = plt.cm.get_cmap("tab20", len(shown_cls))
    colours = [mcolors.to_hex(cmap(i)) for i in range(len(shown_cls))]

    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#161625")

    shown_set = set(shown_cls.tolist())
    mask_other = np.array([lbl not in shown_set for lbl in labels])

    # Plot non-highlighted classes as grey background.
    if mask_other.any():
        ax.scatter(
            tsne_xy[mask_other, 0],
            tsne_xy[mask_other, 1],
            c="#3a3a5c",
            s=6,
            alpha=0.25,
            edgecolors="none",
            rasterized=True,
        )

    # Plot each shown class with its own colour.
    for i, cls in enumerate(shown_cls):
        mask = labels == cls
        ax.scatter(
            tsne_xy[mask, 0],
            tsne_xy[mask, 1],
            c=colours[i],
            label=f"Class {cls}",
            s=12,
            alpha=0.75,
            edgecolors="none",
            rasterized=True,
        )

    ax.set_title(title, color="white", fontsize=14, pad=14, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", color="#aaaacc", fontsize=10)
    ax.set_ylabel("t-SNE dim 2", color="#aaaacc", fontsize=10)
    ax.tick_params(colors="#666688", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

    legend = ax.legend(
        loc="upper right",
        markerscale=2,
        fontsize=7,
        framealpha=0.25,
        labelcolor="white",
        facecolor="#0f0f1a",
        edgecolor="#333355",
        title=f"{len(shown_cls)}/{num_classes} classes shown",
        title_fontsize=7,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_and_run(
    name: str,
    model: torch.nn.Module,
    test_loader,
    device: torch.device,
    use_amp: bool,
    out_dir: Path,
    title: str,
    num_classes: int,
) -> dict:
    """Extract embeddings, run t-SNE, save plot and raw arrays."""
    emb, lbl = extract_embeddings(model, test_loader, device, use_amp, name=name)

    # Cache raw embeddings for later comparison plots.
    np.save(out_dir / f"emb_{name}.npy", emb)
    np.save(out_dir / f"labels_{name}.npy", lbl)

    tsne_xy = run_tsne(emb, perplexity=30, n_iter=1000)
    np.save(out_dir / f"tsne_{name}.npy", tsne_xy)

    plot_tsne(tsne_xy, lbl, title, out_dir / f"tsne_{name}.png", num_classes=num_classes)

    return {
        "name": name,
        "n_samples": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
    }


def main() -> None:
    config = get_config(description="t-SNE Embedding Visualizer — Track 6 Extra Objective")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("training", {}).get("mixed_precision", True)
    num_classes = config["dataset"].get("num_classes", 101)
    width_mult = config.get("model", {}).get("width_mult", 1.0)

    # Output directory
    out_dir = Path(
        os.environ.get(
            "TSNE_LOG_DIR",
            config.get("evaluation", {}).get("tsne_dir", "experiments/logs/tsne"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[t-SNE] Output directory: {out_dir}")

    # Data (test set only for embedding extraction)
    print("[t-SNE] Loading test dataset...")
    dataloaders = get_dataloaders(config)
    test_loader = dataloaders["test"]
    print(f"[t-SNE] Test batches: {len(test_loader)}")

    eval_cfg = config.get("evaluation", {})
    teacher_ckpt: Optional[str] = eval_cfg.get("teacher_checkpoint")
    baseline_ckpt: Optional[str] = eval_cfg.get("baseline_checkpoint")
    distilled_ckpt: Optional[str] = eval_cfg.get("distilled_checkpoint")

    summary: list[dict] = []

    # --- Teacher ---
    if teacher_ckpt:
        print(f"\n[t-SNE] Processing teacher  checkpoint: {teacher_ckpt}")
        teacher = get_teacher(
            num_classes=num_classes,
            pretrained=False,
            checkpoint_path=teacher_ckpt,
        ).to(device)
        info = _load_and_run(
            "teacher", teacher, test_loader, device, use_amp, out_dir,
            title="Teacher — 3D ResNet-50 Embedding Space",
            num_classes=num_classes,
        )
        summary.append(info)
        del teacher
        torch.cuda.empty_cache()

    # --- Baseline student ---
    if baseline_ckpt:
        print(f"\n[t-SNE] Processing baseline checkpoint: {baseline_ckpt}")
        baseline = get_student(
            num_classes=num_classes,
            width_mult=width_mult,
            checkpoint_path=baseline_ckpt,
        ).to(device)
        info = _load_and_run(
            "baseline", baseline, test_loader, device, use_amp, out_dir,
            title="Baseline Student — MobileNet3D (no KD) Embedding Space",
            num_classes=num_classes,
        )
        summary.append(info)
        del baseline
        torch.cuda.empty_cache()

    # --- Distilled student ---
    if distilled_ckpt:
        print(f"\n[t-SNE] Processing distilled checkpoint: {distilled_ckpt}")
        distilled = get_student(
            num_classes=num_classes,
            width_mult=width_mult,
            checkpoint_path=distilled_ckpt,
        ).to(device)
        info = _load_and_run(
            "distilled", distilled, test_loader, device, use_amp, out_dir,
            title="Distilled Student — MobileNet3D (KD) Embedding Space",
            num_classes=num_classes,
        )
        summary.append(info)
        del distilled
        torch.cuda.empty_cache()

    # --- AT Distilled student ---
    at_distilled_ckpt: Optional[str] = eval_cfg.get("at_distilled_checkpoint")
    if at_distilled_ckpt:
        print(f"\n[t-SNE] Processing KD+AT distilled checkpoint: {at_distilled_ckpt}")
        at_distilled = get_student(
            num_classes=num_classes,
            width_mult=width_mult,
            checkpoint_path=at_distilled_ckpt,
        ).to(device)
        info = _load_and_run(
            "distilled_at", at_distilled, test_loader, device, use_amp, out_dir,
            title="Distilled Student — MobileNet3D (KD + Attention Transfer) Embedding Space",
            num_classes=num_classes,
        )
        summary.append(info)
        del at_distilled
        torch.cuda.empty_cache()

    # --- Final summary JSON ---
    summary_path = out_dir / "tsne_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[t-SNE] Summary written to: {summary_path}")
    print("[t-SNE] Done.")


if __name__ == "__main__":
    main()
