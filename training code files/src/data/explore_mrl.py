"""
explore_mrl.py
──────────────
STEP 1: Run this FIRST before any training.

Validates the folder-based MRL dataset and reports:
  - Image counts per split and class
  - Class balance check
  - Image dimensions
  - Corrupted file check
  - Sample grid image
  - Distribution chart

Usage:
    python src/data/explore_mrl.py --root mrl_dataset
"""

import sys
import argparse
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}
SPLITS     = ["train", "val", "test"]
CLASSES    = ["awake", "sleepy"]
LABEL_MAP  = {"awake": 1, "sleepy": 0}
COLOR_MAP  = {"awake": "#2ecc71", "sleepy": "#e74c3c"}


# ──────────────────────────────────────────────────────────────────
# SCAN
# ──────────────────────────────────────────────────────────────────
def scan_all(root: Path) -> dict:
    """
    Returns nested dict:
      data[split][class_name] = list of Path objects
    """
    data = {}

    for split in SPLITS:
        split_dir = root / split
        if not split_dir.exists():
            print(f"  ⚠️  Missing split folder: {split_dir}")
            continue

        data[split] = {}
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                print(f"  ⚠️  Missing class folder: {cls_dir}")
                data[split][cls] = []
                continue

            paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
            data[split][cls] = paths

    return data


# ──────────────────────────────────────────────────────────────────
# PRINT STATS
# ──────────────────────────────────────────────────────────────────
def print_count_table(data: dict):
    print("\n" + "="*55)
    print("  IMAGE COUNTS PER SPLIT & CLASS")
    print("="*55)
    print(f"  {'Split':<8}  {'Awake':>10}  {'Sleepy':>10}  {'Total':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

    grand_total = 0
    for split in SPLITS:
        if split not in data:
            continue
        n_awake  = len(data[split].get("awake",  []))
        n_sleepy = len(data[split].get("sleepy", []))
        n_total  = n_awake + n_sleepy
        grand_total += n_total
        print(f"  {split:<8}  {n_awake:>10,}  {n_sleepy:>10,}  {n_total:>10,}")

    print(f"  {'TOTAL':<8}  {'':>10}  {'':>10}  {grand_total:>10,}")

    # Balance check on TRAIN set
    if "train" in data:
        n_a = len(data["train"].get("awake",  []))
        n_s = len(data["train"].get("sleepy", []))
        if n_a > 0 and n_s > 0:
            ratio = max(n_a, n_s) / min(n_a, n_s)
            if ratio > 1.5:
                minority = "sleepy" if n_s < n_a else "awake"
                print(f"\n  ⚠️  Class imbalance in train set (ratio={ratio:.2f}x).")
                print(f"     '{minority}' is the minority class.")
                print(f"     → WeightedRandomSampler is enabled to compensate.")
            else:
                print(f"\n  ✅  Classes are well-balanced in train (ratio={ratio:.2f}x).")


def check_image_sizes(data: dict, sample_n: int = 200):
    print("\n" + "="*55)
    print(f"  IMAGE SIZE AUDIT  (sampling up to {sample_n} per class)")
    print("="*55)

    import random
    all_sizes = []
    corrupt   = []

    for split in SPLITS:
        if split not in data:
            continue
        for cls in CLASSES:
            paths  = data[split].get(cls, [])
            sample = random.sample(paths, min(sample_n, len(paths)))
            for p in sample:
                try:
                    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        corrupt.append(p)
                    else:
                        all_sizes.append(img.shape)
                except Exception:
                    corrupt.append(p)

    if corrupt:
        print(f"  ⚠️  {len(corrupt)} corrupted/unreadable files found!")
        for c in corrupt[:3]:
            print(f"     {c}")
    else:
        print(f"  ✅  No corrupted files found in sample.")

    if all_sizes:
        heights = [s[0] for s in all_sizes]
        widths  = [s[1] for s in all_sizes]
        size_counter = Counter(all_sizes)

        print(f"\n  Height: min={min(heights)}, max={max(heights)}, avg={np.mean(heights):.1f}")
        print(f"  Width : min={min(widths)}, max={max(widths)}, avg={np.mean(widths):.1f}")
        print(f"\n  Most common sizes (H×W):")
        for size, count in size_counter.most_common(5):
            print(f"    {size[0]:3d}×{size[1]:3d} — {count} images")
        print(f"\n  ℹ️  All images will be resized to 224×224 during training.")


# ──────────────────────────────────────────────────────────────────
# VISUALIZATIONS
# ──────────────────────────────────────────────────────────────────
def visualize_samples(data: dict, save_path: Path, n_per_class: int = 8):
    """Grid: top row = awake, bottom row = sleepy, across all splits."""
    import random

    samples = []
    for cls in CLASSES:
        all_paths = []
        for split in SPLITS:
            all_paths.extend(data.get(split, {}).get(cls, []))
        picked = random.sample(all_paths, min(n_per_class, len(all_paths)))
        samples.extend([(p, cls) for p in picked])

    cols = n_per_class
    rows = len(CLASSES)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 2.0))

    for row_idx, cls in enumerate(CLASSES):
        cls_samples = [(p, c) for p, c in samples if c == cls]
        for col_idx in range(cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(cls_samples):
                img = cv2.imread(str(cls_samples[col_idx][0]), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap="gray")
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(cls.upper(), fontsize=10, fontweight="bold",
                              color=COLOR_MAP[cls], rotation=90, labelpad=6)

    fig.suptitle("MRL Eye Dataset — Sample Images  (top: awake | bottom: sleepy)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  💾  Sample grid → {save_path}")


def visualize_distribution(data: dict, save_path: Path):
    """Grouped bar chart of class counts per split."""
    splits    = [s for s in SPLITS if s in data]
    awake_ns  = [len(data[s].get("awake",  [])) for s in splits]
    sleepy_ns = [len(data[s].get("sleepy", [])) for s in splits]

    x   = np.arange(len(splits))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))

    bars_a = ax.bar(x - w/2, awake_ns,  w, label="Awake",  color="#2ecc71", edgecolor="white")
    bars_s = ax.bar(x + w/2, sleepy_ns, w, label="Sleepy", color="#e74c3c", edgecolor="white")

    for bar in list(bars_a) + list(bars_s):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 50,
                f"{h:,}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_ylabel("Number of Images")
    ax.set_title("MRL Eye Dataset — Class Distribution per Split", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  💾  Distribution chart → {save_path}")


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Explore MRL Eye Dataset (folder-label version)")
    parser.add_argument("--root",   required=True, help="Path to mrl_dataset/ folder")
    parser.add_argument("--outdir", default="outputs/plots", help="Where to save plots")
    args = parser.parse_args()

    root   = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"❌  Dataset root not found: {root}")
        print(f"    Make sure 'mrl_dataset/' exists in your project folder.")
        sys.exit(1)

    print("\n" + "="*55)
    print("  MRL EYE DATASET EXPLORER")
    print("="*55)
    print(f"  Root   : {root.resolve()}")

    # ── Verify structure ──────────────────────────────────────
    print("\n  Expected structure:")
    for split in SPLITS:
        for cls in CLASSES:
            path = root / split / cls
            status = "✅" if path.exists() else "❌  MISSING"
            print(f"    {status}  {root.name}/{split}/{cls}/")

    # ── Scan ──────────────────────────────────────────────────
    data = scan_all(root)

    total = sum(
        len(paths)
        for split in data.values()
        for paths in split.values()
    )
    if total == 0:
        print("\n❌  No images found! Check folder names and image extensions.")
        sys.exit(1)

    print(f"\n  Found {total:,} total images\n")

    # ── Stats ─────────────────────────────────────────────────
    print_count_table(data)
    check_image_sizes(data)

    # ── Visualize ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  GENERATING VISUALIZATIONS")
    print("="*55)
    visualize_samples(data,      save_path=outdir / "mrl_samples.png")
    visualize_distribution(data, save_path=outdir / "mrl_distribution.png")

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*55)
    print("  ✅  EXPLORATION COMPLETE")
    print("="*55)
    print(f"  Total images: {total:,}")
    print()
    print("  NEXT STEP 1 — Sanity check data pipeline:")
    print("    python src/data/mrl_dataset.py --root mrl_dataset")
    print()
    print("  NEXT STEP 2 — Start training:")
    print("    python src/training/train_eye.py --config configs/config.yaml")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
