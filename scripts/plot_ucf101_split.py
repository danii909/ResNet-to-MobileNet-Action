from __future__ import annotations

import argparse
import sys
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


UCF101_SPLITS_URL = (
    "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
)
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "ucf101_splits"


def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"Downloading official UCF101 split annotations from {url} ...")
    context = ssl._create_unverified_context()
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, context=context) as response, dest.open("wb") as file_handle:
        shutil.copyfileobj(response, file_handle)


def _ensure_annotation_dir(cache_dir: Path) -> Path:
    annotation_dir = cache_dir / "ucfTrainTestlist"
    if annotation_dir.exists() and (annotation_dir / "trainlist01.txt").exists():
        return annotation_dir

    archive_path = cache_dir / "UCF101TrainTestSplits-RecognitionTask.zip"
    _download_file(UCF101_SPLITS_URL, archive_path)

    print(f"Extracting split annotations into {cache_dir} ...")
    with zipfile.ZipFile(archive_path, "r") as zip_file:
        zip_file.extractall(cache_dir)

    return annotation_dir


def _count_entries(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file_handle:
        return sum(1 for line in file_handle if line.strip())


def build_split_summary(eval_ratio: float = 0.2, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict[str, dict[str, int]]:
    annotation_dir = _ensure_annotation_dir(cache_dir)

    train_total = _count_entries(annotation_dir / "trainlist01.txt")
    test_total = _count_entries(annotation_dir / "testlist01.txt")

    eval_total = int(round(train_total * eval_ratio))
    train_total_after_split = train_total - eval_total

    summary = {
        "train": {"clips": train_total_after_split},
        "eval": {"clips": eval_total},
        "test": {"clips": test_total},
    }
    summary["totals"] = {"clips": train_total_after_split + eval_total + test_total}
    return summary


def plot_split(summary: dict[str, dict[str, int]], output_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    labels = ["Train", "Eval", "Test"]
    clip_values = [summary[key]["clips"] for key in ("train", "eval", "test")]

    colors = ["#0f766e", "#f59e0b", "#2563eb"]
    accent = "#1f2937"

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 5.0), dpi=140)

    def _annotate_bars(ax, bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(height * 0.015, 4),
                f"{int(height):,}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=accent,
            )

    bars = ax.bar(labels, clip_values, color=colors, edgecolor="none", width=0.6)
    _annotate_bars(ax, bars)
    ax.set_ylim(0, max(clip_values) * 1.18)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot the UCF101 split used in this repo.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "figures" / "ucf101_split_official.png",
        help="Path where the plot will be saved.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="Eval ratio used on the official train split.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Local cache for the official UCF101 split annotations.",
    )
    args = parser.parse_args()

    summary = build_split_summary(eval_ratio=args.eval_ratio, cache_dir=args.cache_dir)
    plot_split(summary, args.output)

    print("Saved plot to:", args.output)
    print("Summary:")
    for key in ("train", "eval", "test"):
        print(f"  {key}: {summary[key]['clips']} videos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())