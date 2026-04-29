"""
verify_crops.py
───────────────
STEP 3 — Run after extract_frames.py.

Checks the output and saves a visual grid so you can confirm:
  - Crops actually show mouth regions (not black frames, not foreheads)
  - Yawn class genuinely looks like open mouths
  - Talking class genuinely looks like closed/slightly-open mouths
  - Class balance is acceptable

Usage:
    python src/verify_crops.py --outdir yawdd_processed
"""

import argparse
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_images(outdir: Path) -> dict:
    counts = {}
    for split in ["train", "val", "test"]:
        for label in ["yawn", "talking"]:
            d = outdir / split / label
            if d.exists():
                n = sum(1 for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                counts[(split, label)] = n
            else:
                counts[(split, label)] = 0
    return counts


def make_sample_grid(outdir: Path, save_path: Path, n_per_class: int = 10):
    """
    Grid layout:
      Row 0: yawn samples
      Row 1: talking samples
    """
    samples = {}
    for label in ["yawn", "talking"]:
        all_paths = []
        for split in ["train", "val", "test"]:
            d = outdir / split / label
            if d.exists():
                all_paths.extend(p for p in d.iterdir()
                                 if p.suffix.lower() in IMAGE_EXTS)
        if all_paths:
            samples[label] = random.sample(all_paths, min(n_per_class, len(all_paths)))
        else:
            samples[label] = []

    cols   = n_per_class
    rows   = 2
    colors = {"yawn": "#e74c3c", "talking": "#2ecc71"}

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))

    for row_i, label in enumerate(["yawn", "talking"]):
        for col_i in range(cols):
            ax = axes[row_i, col_i]
            ax.axis("off")
            if col_i < len(samples[label]):
                img_bgr = cv2.imread(str(samples[label][col_i]))
                if img_bgr is not None:
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
            if col_i == 0:
                ax.set_ylabel(label.upper(), fontsize=9,
                              fontweight="bold", color=colors[label])

    fig.suptitle("YawDD crops — top: YAWN   bottom: TALKING\n"
                 "Verify mouths are correctly cropped and labeled",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grid saved → {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True, help="Path to yawdd_processed/")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        print(f"ERROR: {outdir} not found. Run extract_frames.py first.")
        return

    print("\n" + "="*55)
    print("  YawDD CROP VERIFICATION")
    print("="*55)

    counts = count_images(outdir)

    print(f"\n  {'Split':<8} {'Label':<10} {'Count':>8}")
    print(f"  {'-'*8} {'-'*10} {'-'*8}")
    for split in ["train", "val", "test"]:
        for label in ["yawn", "talking"]:
            n = counts[(split, label)]
            print(f"  {split:<8} {label:<10} {n:>8,}")

    # Balance check
    train_yawn    = counts[("train", "yawn")]
    train_talking = counts[("train", "talking")]
    total_train   = train_yawn + train_talking

    print(f"\n  Train balance:")
    if total_train > 0:
        print(f"    Yawn    : {train_yawn:,}  ({100*train_yawn/total_train:.1f}%)")
        print(f"    Talking : {train_talking:,}  ({100*train_talking/total_train:.1f}%)")
        ratio = max(train_yawn, train_talking) / max(1, min(train_yawn, train_talking))
        if ratio > 3.0:
            print(f"\n  WARNING: Severe imbalance ({ratio:.1f}x).")
            print(f"    Adjust --mar_thresh in extract_frames.py and re-run.")
        elif ratio > 1.5:
            print(f"\n  MILD imbalance ({ratio:.1f}x) — WeightedRandomSampler will handle.")
        else:
            print(f"\n  Good balance ({ratio:.1f}x)!")

    # Visual grid
    plots_dir = outdir / "plots"
    plots_dir.mkdir(exist_ok=True)
    grid_path = plots_dir / "crop_verification.png"
    make_sample_grid(outdir, grid_path)

    print(f"\n  Open {grid_path} to visually check crops.")
    print(f"\n  WHAT TO CHECK:")
    print(f"    YAWN row    → mouths should be WIDE OPEN")
    print(f"    TALKING row → mouths should be closed or slightly open")
    print(f"    Both rows   → crops should be centered on the mouth")
    print(f"    Bad signs   → black frames, cropped eyes/nose, wrong labels")
    print(f"\n  If crops look wrong, adjust --mar_thresh and re-run extraction.")
    print(f"\n  NEXT STEP:")
    print(f"    python src/training/train_yawn.py --config configs/yawn_config.yaml")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
