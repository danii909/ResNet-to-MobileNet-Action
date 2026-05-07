"""Per-class accuracy and confusion matrix analysis.

Extra Objective — Track 6: Knowledge Distillation for Mobile Action Recognition.

Usage (on cluster):
    python -m src.evaluation.confusion \
        --config experiments/configs/distillation_t8_a07_24f_v2.yaml \
        --override evaluation.checkpoint=/path/to/best.pth \
                  evaluation.label=distilled \
                  model.type=student
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.ucf101 import get_dataloaders
from src.models.assistant import get_assistant
from src.models.student import get_student
from src.models.teacher import get_teacher
from src.utils.config import get_config


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full inference and return per-sample predictions and ground truth.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader yielding (clips, labels).
        device: Target device.
        use_amp: Whether to use AMP.

    Returns:
        Tuple of (predictions [N], labels [N]).
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    pbar = tqdm(dataloader, desc="Inference", leave=True, file=sys.stdout)
    for clips, labels in pbar:
        clips = clips.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            logits = model(clips)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.numpy())

        running_acc = (
            np.concatenate(all_preds) == np.concatenate(all_labels)
        ).mean() * 100
        pbar.set_postfix(acc=f"{running_acc:.2f}")

    return np.concatenate(all_preds), np.concatenate(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_per_class_accuracy(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> dict[int, float | None]:
    """Compute per-class top-1 accuracy.

    Returns:
        Dict mapping class index to accuracy percentage, or None if the class
        has no test samples.
    """
    per_class: dict[int, float | None] = {}
    for cls in range(num_classes):
        mask = labels == cls
        if mask.sum() == 0:
            per_class[cls] = None
        else:
            per_class[cls] = float((preds[mask] == cls).mean() * 100)
    return per_class


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Compute and save a normalised confusion matrix as a PNG.

    The confusion matrix is row-normalised so that each row sums to 1,
    making per-class recall patterns visible regardless of class imbalance.

    Args:
        preds: Predicted class indices [N].
        labels: Ground-truth class indices [N].
        num_classes: Total number of classes.
        save_path: Destination PNG file path.
        title: Title for the figure.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # Row-normalise.
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1.0
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall (row-normalised)", fontsize=10)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)

    # Tick labels: show every Nth class to avoid clutter.
    tick_step = max(1, num_classes // 25)
    ticks = list(range(0, num_classes, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks, rotation=90, fontsize=6)
    ax.set_yticklabels(ticks, fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix: {save_path}")


def save_per_class_bar(
    per_class: dict[int, float | None],
    save_path: Path,
    title: str = "Per-class Accuracy",
    top_n: int = 20,
) -> None:
    """Save a horizontal bar chart of the worst and best N classes.

    Args:
        per_class: Dict from class index to accuracy (%).
        save_path: Destination PNG file path.
        title: Figure title.
        top_n: Number of worst/best classes to show.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [(cls, acc) for cls, acc in per_class.items() if acc is not None]
    valid.sort(key=lambda x: x[1])

    worst = valid[:top_n]
    best = valid[-top_n:]
    combined = worst + [("...", None)] + best
    classes_labels = [str(c) for c, _ in combined]
    accs = [a if a is not None else 0 for _, a in combined]
    colours = ["#e05c5c"] * top_n + ["#999999"] + ["#5ce085"] * top_n

    fig, ax = plt.subplots(figsize=(10, max(6, len(combined) * 0.3)))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#161625")
    bars = ax.barh(classes_labels, accs, color=colours, edgecolor="none")
    ax.set_xlabel("Top-1 Accuracy (%)", color="#aaaacc")
    ax.set_title(title, color="white", fontsize=12, fontweight="bold")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.set_xlim(0, 105)

    for bar, acc in zip(bars, accs):
        if acc > 0:
            ax.text(
                acc + 1, bar.get_y() + bar.get_height() / 2,
                f"{acc:.0f}%", va="center", ha="left", color="white", fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved per-class bar chart: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config = get_config(description="Confusion Matrix Analyzer — Track 6 Extra Objective")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.get("training", {}).get("mixed_precision", True)
    num_classes = config["dataset"].get("num_classes", 101)

    out_dir = Path(
        os.environ.get(
            "CONFUSION_LOG_DIR",
            config.get("evaluation", {}).get("confusion_dir", "experiments/logs/confusion"),
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Confusion] Output directory: {out_dir}")

    # Data
    dataloaders = get_dataloaders(config)
    test_loader = dataloaders["test"]
    print(f"[Confusion] Test batches: {len(test_loader)}")

    eval_cfg = config.get("evaluation", {})
    model_type = config["model"]["type"]
    checkpoint = eval_cfg.get("checkpoint")
    label = eval_cfg.get("label", model_type)

    if checkpoint is None:
        raise ValueError("Must set evaluation.checkpoint via config or --override.")

    print(f"[Confusion] Model type: {model_type}, label: {label}")
    print(f"[Confusion] Checkpoint: {checkpoint}")

    # Load model
    if model_type == "teacher":
        model = get_teacher(num_classes=num_classes, pretrained=False, checkpoint_path=checkpoint)
    elif model_type == "assistant":
        model = get_assistant(num_classes=num_classes, pretrained=False, checkpoint_path=checkpoint)
    elif model_type == "student":
        model = get_student(
            num_classes=num_classes,
            width_mult=config.get("model", {}).get("width_mult", 1.0),
            checkpoint_path=checkpoint,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # Inference
    print("[Confusion] Running inference on test set...")
    preds, labels_arr = predict_all(model, test_loader, device, use_amp)

    top1 = float((preds == labels_arr).mean() * 100)
    _, top5_idx = torch.tensor(
        np.zeros((len(preds), num_classes))
    ).topk(5, dim=1)  # placeholder; compute properly below
    # Recompute top-5 using stored logits not available here — report top-1 only.
    print(f"\n[Confusion] Top-1 Accuracy: {top1:.2f}%")

    # Per-class accuracy
    per_class = compute_per_class_accuracy(preds, labels_arr, num_classes)
    valid_accs = [a for a in per_class.values() if a is not None]
    mean_class_acc = float(np.mean(valid_accs)) if valid_accs else 0.0
    print(f"[Confusion] Mean class accuracy: {mean_class_acc:.2f}%")

    sorted_cls = sorted(
        [(cls, acc) for cls, acc in per_class.items() if acc is not None],
        key=lambda x: x[1],
    )
    print("\n  Worst 10 classes:")
    for cls, acc in sorted_cls[:10]:
        print(f"    class {cls:3d}: {acc:.1f}%")
    print("\n  Best 10 classes:")
    for cls, acc in sorted_cls[-10:]:
        print(f"    class {cls:3d}: {acc:.1f}%")

    # Save results
    results = {
        "label": label,
        "model_type": model_type,
        "checkpoint": checkpoint,
        "top1": round(top1, 4),
        "mean_class_accuracy": round(mean_class_acc, 4),
        "per_class_accuracy": {str(cls): acc for cls, acc in per_class.items()},
        "worst_10_classes": [(int(cls), round(acc, 2)) for cls, acc in sorted_cls[:10]],
        "best_10_classes": [(int(cls), round(acc, 2)) for cls, acc in sorted_cls[-10:]],
    }

    json_path = out_dir / f"per_class_{label}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved per-class JSON: {json_path}")

    # Plots
    save_confusion_matrix(
        preds, labels_arr, num_classes,
        save_path=out_dir / f"confusion_{label}.png",
        title=f"Confusion Matrix — {label} (Top-1: {top1:.2f}%)",
    )
    save_per_class_bar(
        per_class,
        save_path=out_dir / f"per_class_bar_{label}.png",
        title=f"Per-class Accuracy — {label} (worst & best 20 classes)",
    )

    print(f"\n[Confusion] Done. All results in: {out_dir}")


if __name__ == "__main__":
    main()
