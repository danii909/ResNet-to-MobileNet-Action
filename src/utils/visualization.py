"""Visualization utilities: t-SNE plots for embedding analysis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract pre-classifier embeddings from a model.

    Args:
        model: Model with a get_embedding() method.
        dataloader: DataLoader to extract from.
        device: Torch device.
        max_samples: Max number of samples to extract (for t-SNE speed).

    Returns:
        Tuple of (embeddings [N, D], labels [N]).
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    count = 0

    for clips, labels in tqdm(dataloader, desc="Extracting embeddings", leave=False):
        if count >= max_samples:
            break
        clips = clips.to(device, non_blocking=True)
        embeddings = model.get_embedding(clips)
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.append(labels.numpy())
        count += clips.size(0)

    embeddings = np.concatenate(all_embeddings, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]
    return embeddings, labels


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    title: str = "t-SNE Visualization",
    perplexity: float = 30.0,
    n_iter: int = 1000,
    max_classes: int = 20,
) -> None:
    """Generate and save a t-SNE scatter plot colored by class.

    Args:
        embeddings: Feature array [N, D].
        labels: Integer labels [N].
        save_path: Path to save the figure.
        title: Plot title.
        perplexity: t-SNE perplexity.
        n_iter: Number of t-SNE iterations.
        max_classes: Max classes to show (for readability).
    """
    # Optionally filter to top-N most frequent classes
    unique_classes, counts = np.unique(labels, return_counts=True)
    if len(unique_classes) > max_classes:
        top_classes = unique_classes[np.argsort(-counts)[:max_classes]]
        mask = np.isin(labels, top_classes)
        embeddings = embeddings[mask]
        labels = labels[mask]

    print(f"Running t-SNE on {len(embeddings)} samples...")
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    coords = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("husl", n_colors=len(np.unique(labels)))
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1],
        hue=labels, palette=palette,
        s=15, alpha=0.7, legend="full",
    )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=2, fontsize=7)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"t-SNE plot saved to {save_path}")


def plot_tsne_comparison(
    models: dict[str, nn.Module],
    dataloader: DataLoader,
    device: torch.device,
    save_dir: str = "figures",
    max_samples: int = 2000,
) -> None:
    """Generate side-by-side t-SNE plots for multiple models.

    Args:
        models: Dict of {model_name: model_instance}.
        dataloader: Test DataLoader.
        device: Torch device.
        save_dir: Directory to save plots.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        embeddings, labels = extract_embeddings(model, dataloader, device, max_samples)
        plot_tsne(
            embeddings, labels,
            save_path=str(save_dir / f"tsne_{name}.png"),
            title=f"t-SNE — {name}",
        )
