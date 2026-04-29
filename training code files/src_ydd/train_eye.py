"""
train_eye.py (YDD Edition)
────────────
Full training pipeline for the YawDD Mouth/Yawn Classifier.
Optimized for CPU training on i5-13500H.
"""

import os
import sys
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ── Project imports ────────────────────────────────────────────────
# Logic to ensure Python can see both the local folder and the 'src' folder
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

sys.path.insert(0, str(current_dir))   # Local folder (for ydd_dataset.py)
sys.path.insert(0, str(project_root))  # Root folder (for src.models)

import ydd_dataset
from ydd_dataset import build_dataloaders
from src.models.eye_model import EyeStateClassifier, build_criterion


# ──────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        print("  Device: CPU  (training will be slower on i5-13500H)")
    return dev


# ──────────────────────────────────────────────────────────────────
# ONE EPOCH: TRAIN
# ──────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp, epoch):
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    pbar = tqdm(loader, desc=f"  Train E{epoch:02d}", leave=False, ncols=90)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return {"loss": total_loss / len(loader.dataset), "accuracy": accuracy_score(all_labels, all_preds)}


# ──────────────────────────────────────────────────────────────────
# ONE EPOCH: EVALUATE
# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device, split="val"):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for images, labels in tqdm(loader, desc=f"  {split.capitalize():5s}     ", leave=False, ncols=90):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    # Using pos_label=1 because 'yawn' is class 1
    prec, rec, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="binary", pos_label=1, zero_division=0)
    
    return {"loss": avg_loss, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "preds": all_preds, "labels": all_labels}


# ──────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────
def save_plots(history, plots_dir, test_metrics=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib
    matplotlib.use("Agg")

    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, [h["train_loss"] for h in history], label="Train")
    plt.plot(epochs, [h["val_loss"] for h in history], label="Val")
    plt.title("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [h["train_acc"] for h in history], label="Train")
    plt.plot(epochs, [h["val_acc"] for h in history], label="Val")
    plt.title("Accuracy"); plt.legend()
    plt.savefig(plots_dir / "training_curves.png")
    
    if test_metrics:
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(test_metrics["labels"], test_metrics["preds"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Talking", "Yawn"], yticklabels=["Talking", "Yawn"])
        plt.title("Confusion Matrix")
        plt.savefig(plots_dir / "confusion_matrix.png")
    plt.close("all")


# ──────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────────
def train(cfg: dict):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*60}\n  STARTING YAWDD RUN: {run_id}\n{'='*60}")

    set_seed(cfg["project"]["seed"])
    device = get_device()
    use_amp = cfg["training"]["amp"] and device.type == "cuda"

    # Dynamic directory creation based on config paths
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / run_id
    plots_dir = Path(cfg["paths"]["plots"]) / run_id
    for d in [ckpt_dir, plots_dir]: d.mkdir(parents=True, exist_ok=True)

    # Building loaders using the logic in ydd_dataset.py
    train_loader, val_loader, test_loader, info = build_dataloaders(
        mrl_root=cfg["paths"]["mrl_root"], 
        img_size=cfg["dataset"]["img_size"],
        batch_size=cfg["training"]["batch_size"], 
        num_workers=cfg["training"]["num_workers"],
        grayscale_to_rgb=cfg["dataset"]["grayscale_to_rgb"]
    )

    model = EyeStateClassifier(
        architecture=cfg["model"]["architecture"], 
        pretrained=cfg["model"]["pretrained"],
        num_classes=cfg["model"]["num_classes"], 
        dropout=cfg["model"]["dropout"]
    ).to(device)

    # Resume logic is commented out to avoid using eye model weights
    # --------------------

    model.freeze_backbone()
    criterion = build_criterion(info["class_counts"], device)
    scaler = GradScaler(enabled=use_amp)

    optimizer = torch.optim.AdamW(model.get_classifier_params(), lr=cfg["training"]["warmup_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg["training"]["scheduler"])

    best_val_loss, patience_count, history = float("inf"), 0, []
    warmup_epochs, total_epochs = cfg["training"]["warmup_epochs"], cfg["training"]["epochs"]
    phase2_switched = False

    for epoch in range(1, total_epochs + 1):
        t_start = time.time()

        if epoch == warmup_epochs + 1 and not phase2_switched:
            print("\n🔓 UNFREEZING ALL LAYERS - PHASE 2 STARTING")
            model.unfreeze_all()
            optimizer = torch.optim.AdamW([
                {"params": model.get_backbone_params(),   "lr": cfg["training"]["full_lr"]},
                {"params": model.get_classifier_params(), "lr": cfg["training"]["warmup_lr"]},
            ], weight_decay=cfg["training"]["weight_decay"])
            
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg["training"]["scheduler"])
            phase2_switched = True

        phase = "warmup" if epoch <= warmup_epochs else "finetune"
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device, "val")
        scheduler.step(val_metrics["loss"])

        print(f"E{epoch:02d} [{phase}] | Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | {time.time()-t_start:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "train_acc": train_metrics["accuracy"], "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss, patience_count = val_metrics["loss"], 0
            torch.save({"model_state": model.state_dict(), "epoch": epoch}, ckpt_dir / "best_model.pth")
        else:
            patience_count += 1
            if patience_count >= cfg["training"]["early_stop_patience"]:
                print("Early stopping triggered."); break

    # Final evaluation on test set
    test_metrics = evaluate(model, test_loader, criterion, device, "test")
    save_plots(history, plots_dir, test_metrics)
    print(f"\n✅ Training Finished. Best Model in: {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()