"""UCF-101 dataset loader with HF frames and optional video decoding."""

import copy
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


def _sample_temporal_indices_with_stride(
    total_frames: int,
    num_frames: int,
    train: bool,
    max_stride: int = 1,
) -> np.ndarray:
    """Sample temporal indices with optional random stride for training.

    When max_stride > 1, train-time clips can span a longer temporal window,
    improving temporal coverage without increasing num_frames.
    """
    if total_frames <= 0:
        raise RuntimeError("Cannot sample from an empty video/clip")

    if max_stride <= 1 or not train:
        return _sample_temporal_indices(total_frames, num_frames, train)

    if total_frames < num_frames:
        return np.arange(num_frames, dtype=int) % total_frames

    stride = int(np.random.randint(1, max_stride + 1))
    needed = (num_frames - 1) * stride + 1

    if needed <= total_frames:
        max_start = total_frames - needed
        start = int(np.random.randint(0, max_start + 1)) if max_start > 0 else 0
        return start + np.arange(num_frames, dtype=int) * stride

    # Fallback to the default segmented sampling when window does not fit.
    return _sample_temporal_indices(total_frames, num_frames, train)


def _apply_clip_spatial_transform(
    frames: torch.Tensor,
    train: bool,
    crop_size: int,
    resize_short_side: int = 128,
    use_random_resized_crop: bool = False,
    rrc_scale: tuple[float, float] = (0.6, 1.0),
    rrc_ratio: tuple[float, float] = (0.75, 1.3333333333),
    color_jitter_strength: float = 0.0,
    random_erasing_prob: float = 0.0,
) -> torch.Tensor:
    """Apply the same spatial transform to all frames in a clip."""
    resized_frames = torch.stack([
        F.resize(frame, [resize_short_side], interpolation=InterpolationMode.BILINEAR)
        for frame in frames
    ])

    if train:
        if use_random_resized_crop:
            crop_i, crop_j, crop_h, crop_w = transforms.RandomResizedCrop.get_params(
                resized_frames[0], scale=rrc_scale, ratio=rrc_ratio
            )
            transformed_frames = [
                F.resized_crop(
                    frame,
                    crop_i,
                    crop_j,
                    crop_h,
                    crop_w,
                    size=[crop_size, crop_size],
                    interpolation=InterpolationMode.BILINEAR,
                )
                for frame in resized_frames
            ]
        else:
            crop_i, crop_j, crop_h, crop_w = transforms.RandomCrop.get_params(
                resized_frames[0], output_size=(crop_size, crop_size)
            )
            transformed_frames = [
                F.crop(frame, crop_i, crop_j, crop_h, crop_w)
                for frame in resized_frames
            ]

        flip = np.random.rand() < 0.5
        if flip:
            transformed_frames = [F.hflip(frame) for frame in transformed_frames]

        # Apply the same color jitter factors to every frame in the clip.
        if color_jitter_strength > 0.0:
            brightness = max(0.0, 1.0 + float(np.random.uniform(-color_jitter_strength, color_jitter_strength)))
            contrast = max(0.0, 1.0 + float(np.random.uniform(-color_jitter_strength, color_jitter_strength)))
            saturation = max(0.0, 1.0 + float(np.random.uniform(-color_jitter_strength, color_jitter_strength)))
            hue_delta = float(np.random.uniform(-0.08, 0.08))
            jittered = []
            for frame in transformed_frames:
                out = F.adjust_brightness(frame, brightness)
                out = F.adjust_contrast(out, contrast)
                out = F.adjust_saturation(out, saturation)
                out = F.adjust_hue(out, hue_delta)
                jittered.append(out)
            transformed_frames = jittered
    else:
        transformed_frames = [
            F.center_crop(frame, [crop_size, crop_size])
            for frame in resized_frames
        ]

    normalized_frames = [
        F.normalize(frame, mean=KINETICS_MEAN, std=KINETICS_STD)
        for frame in transformed_frames
    ]

    if train and random_erasing_prob > 0.0 and np.random.rand() < random_erasing_prob:
        h = normalized_frames[0].shape[1]
        w = normalized_frames[0].shape[2]
        erase_h = max(1, int(h * np.random.uniform(0.02, 0.20)))
        erase_w = max(1, int(w * np.random.uniform(0.02, 0.20)))
        top = int(np.random.randint(0, max(1, h - erase_h + 1)))
        left = int(np.random.randint(0, max(1, w - erase_w + 1)))
        for i in range(len(normalized_frames)):
            normalized_frames[i] = F.erase(
                normalized_frames[i],
                i=top,
                j=left,
                h=erase_h,
                w=erase_w,
                v=0.0,
            )

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

    def clone_for_eval(self) -> "UCF101Dataset":
        """Return a shallow clone that reads the same samples with eval transforms."""
        ds = copy.copy(self)
        ds.train = False
        ds.spatial_transform = ds._build_transforms()
        return ds


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
        resize_short_side: int = 128,
        use_random_resized_crop: bool = False,
        color_jitter_strength: float = 0.0,
        random_erasing_prob: float = 0.0,
        max_temporal_stride: int = 1,
    ):
        from datasets import load_dataset
        from tqdm import tqdm

        self.num_frames = num_frames
        self.crop_size = crop_size
        self.train = train
        self.resize_short_side = resize_short_side
        self.use_random_resized_crop = use_random_resized_crop
        self.color_jitter_strength = color_jitter_strength
        self.random_erasing_prob = random_erasing_prob
        self.max_temporal_stride = max(1, int(max_temporal_stride))

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
            "video_id": self.hf_dataset["video_id"],
            "frame": self.hf_dataset["frame"],
            "label": self.hf_dataset["label"],
            "idx": range(n_total),
        })
        df.sort_values(["clip_id", "frame"], inplace=True)

        self.clips = []
        self.clip_groups = []
        for clip_id, group in df.groupby("clip_id", sort=False):
            self.clips.append((group["idx"].tolist(), int(group["label"].iloc[0])))
            self.clip_groups.append(str(group["video_id"].iloc[0]))

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
            resize_short_side=self.resize_short_side,
            use_random_resized_crop=self.use_random_resized_crop,
            color_jitter_strength=self.color_jitter_strength,
            random_erasing_prob=self.random_erasing_prob,
        )

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row_indices, label = self.clips[idx]

        # Uniformly sample num_frames from the clip
        total = len(row_indices)
        sample_indices = _sample_temporal_indices_with_stride(
            total,
            self.num_frames,
            self.train,
            max_stride=self.max_temporal_stride,
        )

        selected = [row_indices[i] for i in sample_indices]

        # Load frames and apply transforms
        frames = []
        for row_idx in selected:
            img = self.hf_dataset[row_idx]["image"]  # PIL Image
            frames.append(F.to_tensor(img))

        clip = torch.stack(frames)  # [T, C, H, W]
        return self.spatial_transform(clip).permute(1, 0, 2, 3), label

    def clone_for_eval(self) -> "UCF101HFDataset":
        """Return a shallow clone that shares frames/clips but uses eval transforms."""
        ds = copy.copy(self)
        ds.train = False
        ds.max_temporal_stride = 1
        ds.use_random_resized_crop = False
        ds.color_jitter_strength = 0.0
        ds.random_erasing_prob = 0.0
        ds.spatial_transform = ds._build_transforms()
        return ds


def _extract_labels(dataset: Dataset) -> np.ndarray:
    """Extract per-sample labels from dataset internals for stratified splitting."""
    if hasattr(dataset, "samples"):
        # UCF101Dataset: samples = [(path, label), ...]
        return np.asarray([int(lbl) for _, lbl in dataset.samples], dtype=np.int64)
    if hasattr(dataset, "clips"):
        # UCF101HFDataset: clips = [(row_indices, label), ...]
        return np.asarray([int(lbl) for _, lbl in dataset.clips], dtype=np.int64)
    raise ValueError(f"Unsupported dataset type for stratified split: {type(dataset)}")


def _extract_groups(dataset: Dataset) -> np.ndarray | None:
    """Extract per-sample group ids to avoid train/eval leakage.

    For HF clips, groups map to source video_id.
    For file-based UCF101Dataset, each sample is already one video so group==path.
    """
    if hasattr(dataset, "clip_groups"):
        return np.asarray(dataset.clip_groups, dtype=object)
    if hasattr(dataset, "samples"):
        return np.asarray([str(path) for path, _ in dataset.samples], dtype=object)
    return None


def _stratified_sample_split_indices(labels: np.ndarray, eval_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic stratified train/eval indices at sample level."""
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    eval_idx: list[int] = []

    classes = np.unique(labels)
    for cls in classes:
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size == 0:
            continue

        shuffled = cls_idx.copy()
        rng.shuffle(shuffled)

        # At least 1 eval sample if class has >=2 examples.
        n_eval = int(round(shuffled.size * eval_ratio))
        if shuffled.size >= 2:
            n_eval = max(1, min(n_eval, shuffled.size - 1))
        else:
            n_eval = 0

        eval_idx.extend(shuffled[:n_eval].tolist())
        train_idx.extend(shuffled[n_eval:].tolist())

    return np.asarray(train_idx, dtype=np.int64), np.asarray(eval_idx, dtype=np.int64)


def _stratified_group_split_indices(
    labels: np.ndarray,
    groups: np.ndarray,
    eval_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic stratified split while keeping groups disjoint.

    Samples in the same group (video_id) are kept entirely in train or eval.
    """
    if not 0.0 < eval_ratio < 1.0:
        raise ValueError(f"eval_ratio must be in (0, 1), got {eval_ratio}")
    if labels.shape[0] != groups.shape[0]:
        raise ValueError(
            f"labels/groups size mismatch: {labels.shape[0]} vs {groups.shape[0]}"
        )

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    eval_idx: list[int] = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        if cls_idx.size == 0:
            continue

        cls_groups = groups[cls_idx]
        unique_groups = np.unique(cls_groups)

        # If a class has just one group, we cannot make a disjoint group split.
        if unique_groups.size <= 1:
            cls_train, cls_eval = _stratified_sample_split_indices(
                labels[cls_idx], eval_ratio=eval_ratio, seed=seed + int(cls),
            )
            train_idx.extend(cls_idx[cls_train].tolist())
            eval_idx.extend(cls_idx[cls_eval].tolist())
            continue

        group_to_idx: dict[object, np.ndarray] = {}
        for g in unique_groups:
            group_to_idx[g] = cls_idx[cls_groups == g]

        group_order = unique_groups.copy()
        rng.shuffle(group_order)

        target_eval = int(round(cls_idx.size * eval_ratio))
        target_eval = max(1, min(target_eval, cls_idx.size - 1))

        selected_eval_groups: list[object] = []
        eval_count = 0
        remaining = int(cls_idx.size)

        for g in group_order:
            g_count = int(group_to_idx[g].size)
            # Keep at least one sample in train for this class.
            if remaining - g_count < 1:
                continue
            selected_eval_groups.append(g)
            eval_count += g_count
            remaining -= g_count
            if eval_count >= target_eval:
                break

        if eval_count == 0:
            # Last-resort fallback for pathological group distributions.
            cls_train, cls_eval = _stratified_sample_split_indices(
                labels[cls_idx], eval_ratio=eval_ratio, seed=seed + int(cls),
            )
            train_idx.extend(cls_idx[cls_train].tolist())
            eval_idx.extend(cls_idx[cls_eval].tolist())
            continue

        eval_groups_set = set(selected_eval_groups)
        cls_eval_idx = []
        cls_train_idx = []
        for g, g_idx in group_to_idx.items():
            if g in eval_groups_set:
                cls_eval_idx.extend(g_idx.tolist())
            else:
                cls_train_idx.extend(g_idx.tolist())

        train_idx.extend(cls_train_idx)
        eval_idx.extend(cls_eval_idx)

    return np.asarray(train_idx, dtype=np.int64), np.asarray(eval_idx, dtype=np.int64)


def _stratified_split_indices(labels: np.ndarray, eval_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic stratified train/eval indices.

    Ensures at least one sample per class in eval and train whenever class count allows it.
    """
    return _stratified_sample_split_indices(labels, eval_ratio=eval_ratio, seed=seed)


def get_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Create train/eval/test DataLoaders from config.

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

    Optional config keys under 'dataset':
        use_eval_split: if True, split official train set into train/eval (default True).
        eval_ratio: fraction of official train assigned to eval (default 0.2).
        split_seed: seed for deterministic stratified split (default: config['seed'] or 42).
    """
    ds_cfg = config["dataset"]
    tr_cfg = config.get("training", {})

    backend = ds_cfg.get("backend", "hf")
    num_frames = ds_cfg.get("num_frames", 16)
    crop_size = ds_cfg.get("crop_size", 112)
    resize_short_side = ds_cfg.get("resize_short_side", 128)
    use_random_resized_crop = ds_cfg.get("use_random_resized_crop", False)
    color_jitter_strength = ds_cfg.get("color_jitter_strength", 0.0)
    random_erasing_prob = ds_cfg.get("random_erasing_prob", 0.0)
    max_temporal_stride = ds_cfg.get("max_temporal_stride", 1)
    batch_size = tr_cfg.get("batch_size", 16)
    num_workers = tr_cfg.get("num_workers", 4)
    use_eval_split = ds_cfg.get("use_eval_split", True)
    eval_ratio = float(ds_cfg.get("eval_ratio", 0.2))
    split_seed = int(ds_cfg.get("split_seed", config.get("seed", 42)))

    if backend == "hf":
        # HF dataset (flwrlabs/ucf101) - loaded from local disk
        data_dir = ds_cfg.get("data_dir", "data/ucf101")
        print(f"[DataLoaders] Creating HF train dataset (data_dir={data_dir})...")
        train_ds = UCF101HFDataset(
            split="train", num_frames=num_frames,
            crop_size=crop_size, train=True,
            data_dir=data_dir,
            resize_short_side=resize_short_side,
            use_random_resized_crop=use_random_resized_crop,
            color_jitter_strength=color_jitter_strength,
            random_erasing_prob=random_erasing_prob,
            max_temporal_stride=max_temporal_stride,
        )
        eval_source_ds = train_ds.clone_for_eval()
        print(f"[DataLoaders] Creating HF test dataset...")
        test_ds = UCF101HFDataset(
            split="test", num_frames=num_frames,
            crop_size=crop_size, train=False,
            data_dir=data_dir,
            resize_short_side=resize_short_side,
            use_random_resized_crop=False,
            color_jitter_strength=0.0,
            random_erasing_prob=0.0,
            max_temporal_stride=1,
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
        eval_source_ds = train_ds.clone_for_eval()
        test_ds = UCF101Dataset(train=False, **common_kwargs)

    # Build train/eval split from official train split when requested.
    if use_eval_split:
        labels = _extract_labels(train_ds)
        groups = _extract_groups(train_ds)

        if groups is not None:
            train_idx, eval_idx = _stratified_group_split_indices(
                labels, groups, eval_ratio=eval_ratio, seed=split_seed,
            )
        else:
            train_idx, eval_idx = _stratified_split_indices(
                labels, eval_ratio=eval_ratio, seed=split_seed,
            )

        if eval_idx.size == 0 or train_idx.size == 0:
            raise RuntimeError(
                "Invalid train/eval split produced an empty split. "
                f"train={train_idx.size}, eval={eval_idx.size}, eval_ratio={eval_ratio}"
            )

        from torch.utils.data import Subset

        train_ds = Subset(train_ds, train_idx.tolist())
        eval_ds = Subset(eval_source_ds, eval_idx.tolist())

        if groups is not None:
            train_groups = set(groups[train_idx].tolist())
            eval_groups = set(groups[eval_idx].tolist())
            overlap = len(train_groups.intersection(eval_groups))
            print(
                "[DataLoaders] Group split check: "
                f"train_groups={len(train_groups)}, eval_groups={len(eval_groups)}, "
                f"overlap={overlap}"
            )

        print(
            "[DataLoaders] Stratified split enabled: "
            f"train={len(train_ds)} clips, eval={len(eval_ds)} clips, "
            f"ratio={eval_ratio:.2f}, seed={split_seed}"
        )
    else:
        eval_ds = test_ds
        print("[DataLoaders] Stratified eval split disabled: trainer will validate on official test split.")

    print(f"[DataLoaders] Creating DataLoaders (batch_size={batch_size}, workers={num_workers})...")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(
        f"[DataLoaders] Ready: {len(train_loader)} train batches, "
        f"{len(eval_loader)} eval batches, {len(test_loader)} test batches."
    )
    return {"train": train_loader, "eval": eval_loader, "test": test_loader}
