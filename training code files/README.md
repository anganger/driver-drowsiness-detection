# Driver Drowsiness Detection — MRL Dataset Phase
## Step-by-Step Execution Guide

---

## PHASE 1: Environment Setup  (~15 min)

### Step 1.1 — Create a virtual environment

```bash
# Navigate to your project folder
cd drowsiness_detection

# Create venv (Python 3.10+)
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### Step 1.2 — Install all dependencies

```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# ^ Use cu118 for CUDA 11.8. For CPU-only: remove --index-url flag entirely

pip install -r requirements.txt
```

### Step 1.3 — Verify PyTorch + CUDA

```python
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

---

## PHASE 2: Organize your dataset  (~5 min)

### Step 2.1 — Place your MRL dataset

Create a `data/` folder inside the project root and place your dataset there:

```
drowsiness_detection/
├── data/
│   └── mrlEyes_2018_01/        ← unzip your MRL dataset here
│       ├── s0001/
│       │   ├── s0001_00001_0_0_1_0_01_01.png
│       │   ├── s0001_00002_0_0_0_0_01_01.png
│       │   └── ...
│       ├── s0002/
│       └── ...
```

**OR if your dataset is flat (all images in one folder):**
```
data/
└── mrlEyes_2018_01/
    ├── s0001_00001_0_0_1_0_01_01.png
    ├── s0002_00001_0_0_0_0_01_01.png
    └── ...
```

Both layouts are handled automatically.

### Step 2.2 — Update config.yaml

Open `configs/config.yaml` and set the correct path:

```yaml
paths:
  mrl_root: "data/mrlEyes_2018_01"   # ← update this if your folder name differs
```

---

## PHASE 3: Explore the dataset  (~10 min)

**Run this FIRST before any training. It validates your dataset.**

```bash
python src/data/explore_mrl.py --root data/mrlEyes_2018_01
```

This will print:
- Total image count (should be ~85,000)
- Open vs Closed class balance
- Subject count
- Image dimensions (should be 35×55 or 24×36)
- Any corrupted files
- Sample grid saved to `outputs/plots/mrl_samples.png`
- Distribution chart saved to `outputs/plots/mrl_distribution.png`

**What to check:**
- ✅ Total images ~85,000 → dataset loaded correctly
- ✅ Open ~50%, Closed ~50% → balanced (or note imbalance)
- ✅ No corrupted files
- ⚠️ If 0 images found → check the path in --root

---

## PHASE 4: Sanity check the data pipeline  (~5 min)

```bash
python src/data/mrl_dataset.py --root data/mrlEyes_2018_01
```

Expected output:
```
DataLoader sanity check:
  Train batches : 934
  Val batches   : 201
  Test batches  : 201

One batch:
  images.shape  : torch.Size([32, 3, 224, 224])
  images.dtype  : torch.float32
  images range  : [-2.12, 2.64]
  labels        : [1, 0, 1, 0, 1, 1, 0, 0, ...]
```

---

## PHASE 5: Train the model  (~2–8 hours)

```bash
python src/training/train_eye.py --config configs/config.yaml
```

### What happens during training:

**Epochs 1–5 (Warmup/Phase 1):**
- Backbone is FROZEN (ImageNet weights preserved)
- Only the 2-layer classifier head is trained
- LR = 0.001
- Prevents destroying pretrained features with large early gradients

**Epochs 6–30 (Full Fine-tune/Phase 2):**
- All layers UNFROZEN
- Backbone LR = 0.0001 (10× lower — gentle adaptation)
- Head LR = 0.001
- ReduceLROnPlateau: halves LR if val_loss doesn't improve for 4 epochs

**Early stopping:** Stops if val_loss doesn't improve for 8 epochs.

### Training outputs:
```
outputs/
├── checkpoints/<run_id>/
│   └── best_model.pth          ← saved whenever val_loss improves
├── plots/<run_id>/
│   ├── training_curves.png     ← loss + accuracy curves
│   └── confusion_matrix.png    ← test set confusion matrix
└── reports/<run_id>/
    ├── training_log.csv        ← per-epoch metrics
    └── test_report.txt         ← final test results
```

### Expected results:
```
Test Accuracy  : 0.9650+  (target ≥ 0.96)
Test Precision : ~0.97
Test Recall    : ~0.96
Test F1        : ~0.96
Test AUC       : ~0.99
```

---

## PHASE 6: Grad-CAM Visualization  (~5 min)

```bash
python src/utils/gradcam.py \
    --config configs/config.yaml \
    --checkpoint outputs/checkpoints/<run_id>/best_model.pth \
    --split test \
    --n 16
```

Replace `<run_id>` with the timestamp folder created during training (e.g. `20241021_143022`).

This saves a grid showing what the model looks at:
- **Green title** = correct prediction
- **Red title** = incorrect prediction
- Heatmap should highlight **eyelids** for "closed" and **iris/pupil** for "open"

---

## QUICK REFERENCE — Commands in order

```bash
# 1. Explore dataset
python src/data/explore_mrl.py --root data/mrlEyes_2018_01

# 2. Sanity check pipeline
python src/data/mrl_dataset.py --root data/mrlEyes_2018_01

# 3. Train
python src/training/train_eye.py --config configs/config.yaml

# 4. Grad-CAM
python src/utils/gradcam.py --config configs/config.yaml \
    --checkpoint outputs/checkpoints/<run_id>/best_model.pth
```

---

## TROUBLESHOOTING

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `0 images found` | Wrong dataset path | Update `mrl_root` in config.yaml |
| `CUDA out of memory` | batch_size too large | Reduce `batch_size` to 32 or 16 |
| Accuracy stuck at 50% | Labels wrong | Re-run explore_mrl.py, check eye_state field |
| Very slow training | No GPU / too many workers | Set `num_workers: 0`, set `amp: false` |
| Import errors | Wrong working directory | Run all commands from project root |
