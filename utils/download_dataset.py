import os
import argparse
from pathlib import Path
import sys

# Add src to path so we can import our dataset utilities
sys.path.append(str(Path(__file__).parent))

from src.datasets.ucf101 import download_ucf101

def download_hf():
    print("=== Downloading UCF-101 from Hugging Face (flwrlabs/ucf101) ===")
    try:
        from datasets import load_dataset
        print("This might take a while (~2.5M frames)...")
        # Just loading them will trigger the download and caching
        train_ds = load_dataset("flwrlabs/ucf101", split="train")
        print(f"Train split downloaded: {len(train_ds)} frames.")
        test_ds = load_dataset("flwrlabs/ucf101", split="test")
        print(f"Test split downloaded: {len(test_ds)} frames.")
        print("\n[SUCCESS] Dataset cached in Hugging Face directory (usually ~/.cache/huggingface/datasets).")
        print("The experiments are configured to use this 'hf' backend by default.")
    except ImportError:
        print("Error: 'datasets' library not found. Please install it with 'pip install datasets'.")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_raw(data_root):
    print(f"=== Downloading raw UCF-101 to {data_root} ===")
    try:
        video_dir, annotation_dir = download_ucf101(data_root=data_root)
        print(f"\n[SUCCESS] Videos extracted to: {video_dir}")
        print(f"[SUCCESS] Annotations extracted to: {annotation_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download UCF-101 dataset.")
    parser.add_argument("--mode", type=str, choices=["hf", "raw"], default="hf",
                        help="Download mode: 'hf' (Hugging Face frames, default) or 'raw' (Official videos)")
    parser.add_argument("--path", type=str, default="data",
                        help="Destination directory for 'raw' mode (default: data/)")
    
    args = parser.parse_args()

    if args.mode == "hf":
        download_hf()
    else:
        download_raw(args.path)
