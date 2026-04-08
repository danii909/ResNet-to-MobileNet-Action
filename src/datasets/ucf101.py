"""UCF-101 dataset loader with HF frames and optional video decoding."""

import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


# Kinetics mean/std (used by most pretrained video models)
KINETICS_MEAN = [0.45, 0.45, 0.45]
KINETICS_STD = [0.225, 0.225, 0.225]

# Official UCF-101 URLs
_UCF101_VIDEO_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
_UCF101_SPLITS_URL = (
    "https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
)


def _download_file(url: str, dest: Path) -> None:
    """Download a file from url to dest with a progress printout."""
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, str(dest), reporthook=_progress_hook)
    print()  # newline after progress


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


def _sample_temporal_indices(total_frames: int, num_frames: int, train: bool) -> np.ndarray:
    """Sample frame indices with optional temporal jitter."""
    if total_frames <= 0:
        raise RuntimeError("Cannot sample from an empty video/clip")

    if total_frames < num_frames:
        return np.arange(num_frames, dtype=int) % total_frames

    if train:
        boundaries = np.linspace(0, total_frames, num_frames + 1)
        indices = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            start_idx = int(np.floor(start))
            end_idx = int(np.ceil(end))
            end_idx = max(end_idx, start_idx + 1)
            indices.append(np.random.randint(start_idx, min(end_idx, total_frames)))
        return np.asarray(indices, dtype=int)

    return np.linspace(0, total_frames - 1, num_frames, dtype=int)


def _apply_clip_spatial_transform(
    frames: torch.Tensor,
    train: bool,
    crop_size: int,
) -> torch.Tensor:
    """Apply the same spatial transform to all frames in a clip."""
    resized_frames = torch.stack([
        F.resize(frame, [128], interpolation=InterpolationMode.BILINEAR)
        for frame in frames
    ])

    if train:
        crop_i, crop_j, crop_h, crop_w = transforms.RandomCrop.get_params(
            resized_frames[0], output_size=(crop_size, crop_size)
        )
        flip = np.random.rand() < 0.5
        transformed_frames = [
            F.crop(frame, crop_i, crop_j, crop_h, crop_w)
            for frame in resized_frames
        ]
        if flip:
            transformed_frames = [F.hflip(frame) for frame in transformed_frames]
    else:
        transformed_frames = [
            F.center_crop(frame, [crop_size, crop_size])
            for frame in resized_frames
        ]

    normalized_frames = [
        F.normalize(frame, mean=KINETICS_MEAN, std=KINETICS_STD)
        for frame in transformed_frames
    ]
    return torch.stack(normalized_frames)


def download_ucf101(data_root: str = "data") -> tuple[Path, Path]:
    """Download and extract UCF-101 videos and train/test split annotations.

    Directory layout after download:
        data_root/
            UCF-101/          <- video folders (ApplyEyeMakeup/, ...)
            ucfTrainTestlist/  <- classInd.txt, trainlist01.txt, ...

    Args:
        data_root: Root directory for data storage.

    Returns:
        Tuple of (video_dir, annotation_dir) paths.
    """
    root = Path(data_root)
    root.mkdir(parents=True, exist_ok=True)

    video_dir = root / "UCF-101"
    annotation_dir = root / "ucfTrainTestlist"

    # --- Download & extract videos ---
    if not video_dir.exists() or not any(video_dir.iterdir()):
        rar_path = root / "UCF101.rar"
        if not rar_path.exists():
            _download_file(_UCF101_VIDEO_URL, rar_path)

        print("Extracting UCF101.rar (this may take a while) ...")
        # Try unrar first, fall back to python rarfile
        if shutil.which("unrar"):
            subprocess.run(
                ["unrar", "x", "-o+", str(rar_path), str(root)],
                check=True,
            )
        else:
            try:
                import rarfile
                with rarfile.RarFile(str(rar_path)) as rf:
                    rf.extractall(str(root))
            except ImportError:
                raise RuntimeError(
                    "Cannot extract .rar: install 'unrar' system tool or "
                    "'pip install rarfile' (plus unrar backend). "
                    "Alternatively, manually extract UCF101.rar into "
                    f"{root / 'UCF-101'}"
                )
        print(f"Videos extracted to {video_dir}")

        # Clean up archive
        if rar_path.exists():
            rar_path.unlink()
    else:
        print(f"UCF-101 videos already present at {video_dir}")

    # --- Download & extract annotations ---
    if not annotation_dir.exists() or not (annotation_dir / "classInd.txt").exists():
        zip_path = root / "UCF101TrainTestSplits-RecognitionTask.zip"
        if not zip_path.exists():
            _download_file(_UCF101_SPLITS_URL, zip_path)

        print("Extracting annotation splits ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(root))
        print(f"Annotations extracted to {annotation_dir}")

        if zip_path.exists():
            zip_path.unlink()
    else:
        print(f"UCF-101 annotations already present at {annotation_dir}")

    return video_dir, annotation_dir


def _try_import_decord():
    try:
        import decord
        decord.bridge.set_bridge("torch")
        return decord
    except ImportError:
        return None


def _read_video_decord(path: str, num_frames: int, train: bool) -> torch.Tensor:
    """Read video using decord. Returns tensor of shape [T, H, W, C] uint8."""
    import decord
    vr = decord.VideoReader(path, num_threads=1)
    total = len(vr)
    if total == 0:
        raise RuntimeError(f"Empty video: {path}")
    indices = _sample_temporal_indices(total, num_frames, train)
    frames = vr.get_batch(indices)  # [T, H, W, C] torch tensor
    return frames


def _read_video_torchvision(path: str, num_frames: int, train: bool) -> torch.Tensor:
    """Fallback: read video using torchvision (pyav). Returns [T, H, W, C] uint8."""
    from torchvision.io import read_video as tv_read_video
    video, _, info = tv_read_video(path, pts_unit="sec")
    total = video.shape[0]
    if total == 0:
        raise RuntimeError(f"Empty video: {path}")
    indices = _sample_temporal_indices(total, num_frames, train)
    return video[indices]


class UCF101Dataset(Dataset):
    """UCF-101 video dataset.

    Expects the UCF-101 directory structure:
        data_dir/
            ApplyEyeMakeup/
                v_ApplyEyeMakeup_g01_c01.avi
                ...
            ...

    And annotation files:
        annotation_dir/
            trainlist01.txt
            testlist01.txt
            classInd.txt

    Args:
        data_dir: Path to the UCF-101 video folder.
        annotation_dir: Path to the folder with split files.
        split: Which split to use (1, 2, or 3).
        train: If True, use training set; otherwise test set.
        num_frames: Number of frames to uniformly sample per clip.
        crop_size: Spatial crop size (height, width).
        backend: Video decoding backend ('decord' or 'torchvision').
    """

    def __init__(
        self,
        data_dir: str,
        annotation_dir: str,
        split: int = 1,
        train: bool = True,
        num_frames: int = 16,
        crop_size: int = 112,
        backend: str = "decord",
    ):
        self.data_dir = Path(data_dir)
        self.annotation_dir = Path(annotation_dir)
        self.train = train
        self.num_frames = num_frames
        self.crop_size = crop_size

        # Auto-download if data is missing
        if not self.data_dir.exists() or not self.annotation_dir.exists():
            print("Dataset not found. Downloading UCF-101 ...")
            self.data_dir, self.annotation_dir = download_ucf101(
                data_root=str(self.data_dir.parent)
            )

        # Select backend
        if backend == "decord" and _try_import_decord() is not None:
            self._read_video = _read_video_decord
        else:
            self._read_video = _read_video_torchvision

        # Load class mapping
        self.class_to_idx = self._load_class_index()
        self.num_classes = len(self.class_to_idx)

        # Load file list
        split_file = "trainlist{:02d}.txt" if train else "testlist{:02d}.txt"
        split_path = self.annotation_dir / split_file.format(split)
        self.samples = self._parse_split_file(split_path)

        # Transforms
        self.spatial_transform = self._build_transforms()

    def _load_class_index(self) -> dict[str, int]:
        """Parse classInd.txt → {class_name: index (0-based)}."""
        class_file = self.annotation_dir / "classInd.txt"
        class_to_idx = {}
        with open(class_file, "r") as f:
            for line in f:
                idx, name = line.strip().split()
                class_to_idx[name] = int(idx) - 1  # 0-based
        return class_to_idx

    def _parse_split_file(self, path: Path) -> list[tuple[str, int]]:
        """Parse train/test split file. Returns list of (video_path, label)."""
        samples = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                rel_path = parts[0]  # e.g., ApplyEyeMakeup/v_..._.avi
                class_name = rel_path.split("/")[0]
                label = self.class_to_idx[class_name]
                full_path = str(self.data_dir / rel_path)
                samples.append((full_path, label))
        return samples

    def _build_transforms(self):
        """Build clip-level spatial transforms."""
        return lambda frames: _apply_clip_spatial_transform(
            frames=frames,
            train=self.train,
            crop_size=self.crop_size,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]

        # Read video: [T, H, W, C] uint8
        frames = self._read_video(path, self.num_frames, self.train)

        # Convert to float [T, C, H, W] in [0, 1]
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)  # [T, C, H, W]

        return self.spatial_transform(frames).permute(1, 0, 2, 3), label


class UCF101HFDataset(Dataset):
    """UCF-101 dataset loaded from Hugging Face (flwrlabs/ucf101).

    The HF dataset contains individual frames with fields:
        image (PIL), video_id, clip_id, frame, label

    This class groups frames by clip_id and samples num_frames per clip.

    Args:
        split: 'train' or 'test'.
        num_frames: Number of frames to uniformly sample per clip.
        crop_size: Spatial crop size.
        train: If True, use training augmentations.
    """

    def __init__(
        self,
        split: str = "train",
        num_frames: int = 16,
        crop_size: int = 112,
        train: bool = True,
        data_dir: str = "data/ucf101",
    ):
        from datasets import load_dataset
        from tqdm import tqdm

        self.num_frames = num_frames
        self.crop_size = crop_size
        self.train = train

        # Load dataset from HF cache (downloaded by setup.sh, no network needed
        # if HF_DATASETS_OFFLINE=1 is set)
        import time

        t0 = time.time()
        print(f"[Dataset] Loading UCF-101 '{split}' from HF cache...")
        self.hf_dataset = load_dataset("flwrlabs/ucf101", split=split)
        n_total = len(self.hf_dataset)
        print(f"[Dataset] Loaded {n_total} frames in {time.time()-t0:.1f}s")

        # Build clip index using pandas groupby (much faster than Python loop)
        t0 = time.time()
        print("[Dataset] Building clip index with pandas...")
        import pandas as pd

        df = pd.DataFrame({
            "clip_id": self.hf_dataset["clip_id"],
            "frame": self.hf_dataset["frame"],
            "label": self.hf_dataset["label"],
            "idx": range(n_total),
        })
        df.sort_values(["clip_id", "frame"], inplace=True)

        self.clips = []
        for clip_id, group in df.groupby("clip_id", sort=False):
            self.clips.append((group["idx"].tolist(), int(group["label"].iloc[0])))

        self.num_classes = df["label"].nunique()
        print(f"[Dataset] Clip index ready in {time.time()-t0:.1f}s: "
              f"{len(self.clips)} clips, {self.num_classes} classes, "
              f"avg {n_total//len(self.clips):.0f} frames/clip")

        self.spatial_transform = self._build_transforms()

    def _build_transforms(self):
        return lambda frames: _apply_clip_spatial_transform(
            frames=frames,
            train=self.train,
            crop_size=self.crop_size,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row_indices, label = self.clips[idx]

        # Uniformly sample num_frames from the clip
        total = len(row_indices)
        sample_indices = _sample_temporal_indices(total, self.num_frames, self.train)

        selected = [row_indices[i] for i in sample_indices]

        # Load frames and apply transforms
        frames = []
        for row_idx in selected:
            img = self.hf_dataset[row_idx]["image"]  # PIL Image
            frames.append(F.to_tensor(img))

        clip = torch.stack(frames)  # [T, C, H, W]
        return self.spatial_transform(clip).permute(1, 0, 2, 3), label


def get_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Create train and test DataLoaders from config.

    Expected config keys under 'dataset':
        backend: 'hf', 'decord', or 'torchvision' (default 'hf').
        num_frames: Frames per clip (default 16).
        crop_size: Spatial crop size (default 112).
        For 'decord'/'torchvision' backends only:
            data_dir: Path to UCF-101 videos.
            annotation_dir: Path to UCF-101 annotation files.
            split: Split number (default 1).

    Expected config keys under 'training':
        batch_size: Batch size.
        num_workers: DataLoader workers (default 4).
    """
    ds_cfg = config["dataset"]
    tr_cfg = config.get("training", {})

    backend = ds_cfg.get("backend", "hf")
    num_frames = ds_cfg.get("num_frames", 16)
    crop_size = ds_cfg.get("crop_size", 112)
    batch_size = tr_cfg.get("batch_size", 16)
    num_workers = tr_cfg.get("num_workers", 4)

    if backend == "hf":
        # HF dataset (flwrlabs/ucf101) - loaded from local disk
        data_dir = ds_cfg.get("data_dir", "data/ucf101")
        print(f"[DataLoaders] Creating HF train dataset (data_dir={data_dir})...")
        train_ds = UCF101HFDataset(
            split="train", num_frames=num_frames,
            crop_size=crop_size, train=True,
            data_dir=data_dir,
        )
        print(f"[DataLoaders] Creating HF test dataset...")
        test_ds = UCF101HFDataset(
            split="test", num_frames=num_frames,
            crop_size=crop_size, train=False,
            data_dir=data_dir,
        )
    else:
        # Video-file based dataset
        data_dir = Path(ds_cfg["data_dir"])
        annotation_dir = Path(ds_cfg["annotation_dir"])
        if not data_dir.exists() or not annotation_dir.exists():
            data_dir, annotation_dir = download_ucf101(
                data_root=str(data_dir.parent)
            )
            ds_cfg["data_dir"] = str(data_dir)
            ds_cfg["annotation_dir"] = str(annotation_dir)

        common_kwargs = dict(
            data_dir=ds_cfg["data_dir"],
            annotation_dir=ds_cfg["annotation_dir"],
            split=ds_cfg.get("split", 1),
            num_frames=num_frames,
            crop_size=crop_size,
            backend=backend,
        )
        train_ds = UCF101Dataset(train=True, **common_kwargs)
        test_ds = UCF101Dataset(train=False, **common_kwargs)

    print(f"[DataLoaders] Creating DataLoaders (batch_size={batch_size}, workers={num_workers})...")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"[DataLoaders] Ready: {len(train_loader)} train batches, {len(test_loader)} test batches.")
    return {"train": train_loader, "test": test_loader}
