"""
mrl_dataset.py
──────────────
PyTorch Dataset for the MRL Eye Dataset — folder-label version.

Your actual dataset structure:
    mrl_dataset/
        train/
            awake/    ← label 1  (eyes open)
            sleepy/   ← label 0  (eyes closed / drowsy)
        val/
            awake/
            sleepy/
        test/
            awake/
            sleepy/

No filename parsing needed. Labels come entirely from folder names.
The dataset is already pre-split — we do NOT re-split it.
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
# CONSTANTS
# ──────────────────────────────────────────────────────────────────

# Folder name → integer label
FOLDER_TO_LABEL = {
    "awake":  1,   # eyes open
    "sleepy": 0,   # eyes closed / drowsy
}

LABEL_TO_NAME = {1: "awake", 0: "sleepy"}

# ImageNet normalization (required for MobileNetV2 pretrained weights)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


# ──────────────────────────────────────────────────────────────────
# SCAN ONE SPLIT FOLDER  (e.g. mrl_dataset/train)
# ──────────────────────────────────────────────────────────────────
def scan_split_folder(split_dir: Path) -> list[tuple[Path, int]]:
    """
    Walk  split_dir/awake/  and  split_dir/sleepy/
    Return list of (image_path, label) tuples.
    """
    records = []

    for folder_name, label in FOLDER_TO_LABEL.items():
        class_dir = split_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Expected folder not found: {class_dir}\n"
                f"Make sure your dataset has: {split_dir}/awake/ and {split_dir}/sleepy/"
            )
        for p in class_dir.iterdir():
            if p.suffix.lower() in IMAGE_EXTS:
                records.append((p, label))

    return records


# ──────────────────────────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────────────────────────
def get_transforms(
    split:       Literal["train", "val", "test"],
    img_size:    int   = 224,
    rotation:    int   = 15,
    brightness:  float = 0.3,
    contrast:    float = 0.3,
    crop_pad:    int   = 10,
) -> transforms.Compose:
    """
    Train  → resize FIRST, then augmentation + normalize
    Val/Test → resize + normalize only
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            # 1. First, resize the tiny image to be slightly larger than the target
            transforms.Resize((img_size + 32, img_size + 32)), 
            
            # 2. Now it is safe to crop 224x224
            transforms.RandomCrop(img_size),
            
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=rotation),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=0.1,
                hue=0.0,
            ),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            # Val and Test just get a direct resize to the target size
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

# ──────────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────────
class MRLEyeDataset(Dataset):
    """
    Loads images from mrl_dataset/{split}/awake/ and mrl_dataset/{split}/sleepy/

    Each item:
        image  : FloatTensor  [3, H, W]   (ImageNet-normalized)
        label  : int          0=sleepy, 1=awake
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

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((224, 224), dtype=np.uint8)

        # MobileNetV2 expects 3 channels — replicate grayscale across R,G,B
        if self.grayscale_to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        pil_img = Image.fromarray(img)
        tensor  = self.transform(pil_img)
        return tensor, label

    def get_labels(self) -> list[int]:
        return [label for _, label in self.records]

    def class_counts(self) -> torch.Tensor:
        """Return tensor([n_sleepy, n_awake])."""
        labels = torch.tensor(self.get_labels())
        return torch.bincount(labels, minlength=2)


# ──────────────────────────────────────────────────────────────────
# WEIGHTED SAMPLER
# ──────────────────────────────────────────────────────────────────
def make_weighted_sampler(dataset: MRLEyeDataset) -> WeightedRandomSampler:
    """
    Each class gets equal representation per epoch.
    weight[i] = 1 / count(class of sample i)
    """
    labels             = dataset.get_labels()
    counts             = torch.bincount(torch.tensor(labels), minlength=2)
    weights_per_class  = 1.0 / counts.float()
    sample_weights     = weights_per_class[torch.tensor(labels)]

    return WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
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
    # Kept for API compatibility with train_eye.py — not used here
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
    seed:        int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Build train / val / test DataLoaders from the pre-split folder structure.
    Returns: train_loader, val_loader, test_loader, info_dict
    """
    root = Path(mrl_root)

    for split in ["train", "val", "test"]:
        if not (root / split).exists():
            raise FileNotFoundError(
                f"Split folder missing: {root / split}\n"
                f"Expected structure: {root}/train/, {root}/val/, {root}/test/"
            )

    print(f"\n📂  Loading MRL dataset from: {root.resolve()}")

    train_records = scan_split_folder(root / "train")
    val_records   = scan_split_folder(root / "val")
    test_records  = scan_split_folder(root / "test")

    print(f"    train : {len(train_records):,} images")
    print(f"    val   : {len(val_records):,} images")
    print(f"    test  : {len(test_records):,} images")

    train_tf = get_transforms("train", img_size=img_size)
    val_tf   = get_transforms("val",   img_size=img_size)
    test_tf  = get_transforms("test",  img_size=img_size)

    train_ds = MRLEyeDataset(train_records, train_tf, grayscale_to_rgb)
    val_ds   = MRLEyeDataset(val_records,   val_tf,   grayscale_to_rgb)
    test_ds  = MRLEyeDataset(test_records,  test_tf,  grayscale_to_rgb)

    sampler = make_weighted_sampler(train_ds) if use_sampler else None
    shuffle = (sampler is None)

    loader_kwargs = dict(
        num_workers        = num_workers,
        pin_memory         = True,
        persistent_workers = (num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, shuffle=shuffle, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **loader_kwargs,
    )

    info = {
        "n_total":      len(train_records) + len(val_records) + len(test_records),
        "n_train":      len(train_ds),
        "n_val":        len(val_ds),
        "n_test":       len(test_ds),
        "class_counts": train_ds.class_counts(),   # [n_sleepy, n_awake]
        "label_names":  LABEL_TO_NAME,
    }

    return train_loader, val_loader, test_loader, info


# ──────────────────────────────────────────────────────────────────
# SANITY CHECK
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True,
                        help="Path to mrl_dataset/ (the folder containing train/val/test)")
    args = parser.parse_args()

    train_loader, val_loader, test_loader, info = build_dataloaders(
        mrl_root    = args.root,
        batch_size  = 32,
        num_workers = 0,
    )

    print(f"\n  ✅  DataLoader sanity check:")
    print(f"    Train batches : {len(train_loader)}")
    print(f"    Val batches   : {len(val_loader)}")
    print(f"    Test batches  : {len(test_loader)}")
    print(f"    Class counts  : sleepy={info['class_counts'][0]}, awake={info['class_counts'][1]}")

    images, labels = next(iter(train_loader))
    print(f"\n  One batch:")
    print(f"    images.shape : {images.shape}")
    print(f"    value range  : [{images.min():.2f}, {images.max():.2f}]")
    print(f"    labels sample: {labels.tolist()[:16]}")
    print(f"\n  ✅  Pipeline working! Ready to train.")
    print(f"  NEXT: python src/training/train_eye.py --config configs/config.yaml")
