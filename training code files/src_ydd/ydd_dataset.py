"""
ydd_dataset.py
──────────────
Unified PyTorch Dataset for BOTH:
  - MRL Eye Dataset   → folders: awake/ sleepy/
  - YawDD mouth crops → folders: yawn/  talking/

Auto-detects which dataset it's loading by reading the folder names.
"""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


# ──────────────────────────────────────────────────────────────────
# SUPPORTED FOLDER-NAME → LABEL MAPPINGS
# ──────────────────────────────────────────────────────────────────

KNOWN_LABEL_MAPS = [
    # MRL Eye Dataset
    {"awake": 1, "sleepy": 0},
    # YawDD mouth crops
    {"yawn": 1, "talking": 0},
]

LABEL_TO_NAME_MRL   = {1: "awake",   0: "sleepy"}
LABEL_TO_NAME_YAWDD = {1: "yawn",    0: "talking"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ──────────────────────────────────────────────────────────────────
# AUTO-DETECT LABEL MAP FROM FOLDER NAMES
# ──────────────────────────────────────────────────────────────────

def detect_label_map(split_dir: Path) -> dict[str, int]:
    subfolders = {p.name for p in split_dir.iterdir() if p.is_dir()}

    for lmap in KNOWN_LABEL_MAPS:
        if set(lmap.keys()) == subfolders:
            return lmap

    sorted_folders = sorted(subfolders)
    lmap = {name: i for i, name in enumerate(sorted_folders)}
    print(f"  WARNING: Unknown folder names {subfolders}. "
          f"Assigning labels alphabetically: {lmap}")
    return lmap


# ──────────────────────────────────────────────────────────────────
# SCAN ONE SPLIT
# ──────────────────────────────────────────────────────────────────

def scan_split_folder(split_dir: Path, label_map: dict) -> list[tuple[Path, int]]:
    records = []
    for folder_name, label in label_map.items():
        class_dir = split_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Expected folder missing: {class_dir}")
        for p in class_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                records.append((p, label))
    return records


# ──────────────────────────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────────────────────────

def get_transforms(
    split:      Literal["train", "val", "test"],
    img_size:   int   = 224,
    rotation:   int   = 15,
    brightness: float = 0.3,
    contrast:   float = 0.3,
    crop_pad:   int   = 10,
) -> transforms.Compose:
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Pad(crop_pad, padding_mode="reflect"),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation),
            transforms.ColorJitter(brightness=brightness, contrast=contrast,
                                   saturation=0.1, hue=0.0),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])


# ──────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────

class YDDRelatedDataset(Dataset):
    """
    Works for both MRL (eye state) and YawDD (mouth/yawn) crops.
    """
    def __init__(
        self,
        records:          list[tuple[Path, int]],
        transform:        transforms.Compose,
        grayscale_to_rgb: bool = True,
    ):
        self.records          = records
        self.transform        = transform
        self.grayscale_to_rgb = grayscale_to_rgb

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_path, label = self.records[idx]

        if self.grayscale_to_rgb:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 224), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(img)
        tensor  = self.transform(pil_img)
        return tensor, label

    def get_labels(self) -> list[int]:
        return [label for _, label in self.records]

    def class_counts(self) -> torch.Tensor:
        labels = torch.tensor(self.get_labels())
        return torch.bincount(labels, minlength=2)


# ──────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER
# ──────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset: YDDRelatedDataset) -> WeightedRandomSampler:
    labels            = dataset.get_labels()
    counts            = torch.bincount(torch.tensor(labels), minlength=2)
    weights_per_class = 1.0 / counts.float()
    sample_weights    = weights_per_class[torch.tensor(labels)]
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )


# ──────────────────────────────────────────────────────────────────
# DATALOADER FACTORY
# ──────────────────────────────────────────────────────────────────

def build_dataloaders(
    mrl_root:         str | Path,
    img_size:         int  = 224,
    batch_size:       int  = 64,
    num_workers:      int  = 4,
    grayscale_to_rgb: bool = True,
    use_sampler:      bool = True,
    seed:             int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:

    root = Path(mrl_root)
    label_map = detect_label_map(root / "train")
    label_to_name = {v: k for k, v in label_map.items()}

    train_records = scan_split_folder(root / "train", label_map)
    val_records   = scan_split_folder(root / "val",   label_map)
    test_records  = scan_split_folder(root / "test",  label_map)

    train_tf = get_transforms("train", img_size=img_size)
    val_tf   = get_transforms("val",   img_size=img_size)
    test_tf  = get_transforms("test",  img_size=img_size)

    train_ds = YDDRelatedDataset(train_records, train_tf, grayscale_to_rgb)
    val_ds   = YDDRelatedDataset(val_records,   val_tf,   grayscale_to_rgb)
    test_ds  = YDDRelatedDataset(test_records,  test_tf,  grayscale_to_rgb)

    sampler = make_weighted_sampler(train_ds) if use_sampler else None
    shuffle = (sampler is None)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, shuffle=shuffle, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, **loader_kwargs)

    info = {
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "class_counts": train_ds.class_counts(), "label_names": label_to_name,
    }

    return train_loader, val_loader, test_loader, info