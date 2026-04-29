"""
gradcam.py
──────────
Grad-CAM visualizations for the trained Eye State Classifier.

What Grad-CAM does:
  It answers: "Which spatial regions of the input image did the model
  focus on when making its prediction?"

  For an eye image: we expect the model to highlight the eyelid region
  for "closed" predictions and the iris/pupil for "open" predictions.
  If it highlights irrelevant regions (nose, forehead), the model has
  learned shortcuts — a red flag for deployment.

Usage:
    python src/utils/gradcam.py \
        --config configs/config.yaml \
        --checkpoint outputs/checkpoints/<run_id>/best_model.pth \
        --split test
"""

import sys
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.data.mrl_dataset  import build_dataloaders
from src.models.eye_model  import EyeStateClassifier

LABEL_NAMES = {0: "closed", 1: "open"}


# ──────────────────────────────────────────────────────────────────
# MANUAL GRAD-CAM (no external library needed)
# ──────────────────────────────────────────────────────────────────
class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Works for any CNN model. We hook into the last conv layer,
    capture the forward activations and backward gradients,
    then compute a weighted sum to produce the heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.activations  = None
        self.gradients    = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        input_tensor: torch.Tensor,  # [1, 3, H, W]
        class_idx: int | None = None,
    ) -> np.ndarray:
        """
        Returns a heatmap array of shape [H, W] normalized to [0, 1].
        class_idx=None → use predicted class.
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for the target class
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Global average pool of gradients: [C]
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # [1, C, 1, 1]

        # Weighted combination of activation maps: [H', W']
        cam = (weights * self.activations).sum(dim=1).squeeze()   # [H', W']
        cam = torch.relu(cam).cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model: EyeStateClassifier):
    """Return the last convolutional layer for Grad-CAM."""
    arch = model.architecture
    if arch == "mobilenetv2":
        # Last conv block of MobileNetV2
        return model.backbone.features[-1][0]
    elif arch == "efficientnet_b0":
        return model.backbone.features[-1][0]
    else:
        raise ValueError(f"Unknown arch: {arch}")


def tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a normalized image tensor [3, H, W] back to uint8 RGB for display.
    Reverses ImageNet normalization.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
    img  = img * std + mean
    img  = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image."""
    # Resize CAM to image size
    h, w = img_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Apply colormap
    heatmap = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (alpha * heatmap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


# ──────────────────────────────────────────────────────────────────
# VISUALIZATION GRID
# ──────────────────────────────────────────────────────────────────
def visualize_gradcam(
    model:      EyeStateClassifier,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    save_path:  Path,
    n_samples:  int = 16,
):
    """
    Create a grid showing: Original | Grad-CAM Overlay | Prediction
    """
    grad_cam      = GradCAM(model, get_target_layer(model))
    model.eval()

    samples   = []   # (img_tensor, true_label)
    collected = 0

    for images, labels in loader:
        for i in range(images.size(0)):
            if collected >= n_samples:
                break
            samples.append((images[i], labels[i].item()))
            collected += 1
        if collected >= n_samples:
            break

    cols = 4
    rows = (n_samples + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.5))
    axes = np.array(axes).reshape(rows, cols * 2)

    for idx, (img_tensor, true_label) in enumerate(tqdm(samples, desc="  Grad-CAM")):
        row = idx // cols
        col = (idx % cols) * 2

        inp = img_tensor.unsqueeze(0).to(device)
        cam, pred_class = grad_cam(inp)

        img_np  = tensor_to_display(img_tensor)
        overlay = overlay_heatmap(img_np, cam)

        correct = (pred_class == true_label)
        color   = "green" if correct else "red"
        title   = (f"T:{LABEL_NAMES[true_label]} "
                   f"P:{LABEL_NAMES[pred_class]} "
                   f"{'✓' if correct else '✗'}")

        # Original
        axes[row, col].imshow(img_np)
        axes[row, col].axis("off")
        axes[row, col].set_title("Original", fontsize=7)

        # Overlay
        axes[row, col + 1].imshow(overlay)
        axes[row, col + 1].axis("off")
        axes[row, col + 1].set_title(title, fontsize=7, color=color, fontweight="bold")

    fig.suptitle("Grad-CAM — What the Model Looks At", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    grad_cam.remove_hooks()
    print(f"\n  💾  Grad-CAM grid saved → {save_path}")


# ──────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--n",          type=int, default=16)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = EyeStateClassifier(
        architecture = cfg["model"]["architecture"],
        pretrained   = False,
        num_classes  = cfg["model"]["num_classes"],
        dropout      = cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  ✅  Model loaded from {args.checkpoint}")

    # Loaders
    _, val_loader, test_loader, _ = build_dataloaders(
        mrl_root    = cfg["paths"]["mrl_root"],
        img_size    = cfg["dataset"]["img_size"],
        batch_size  = 1,   # one image at a time for Grad-CAM
        num_workers = 0,
        seed        = cfg["project"]["seed"],
    )
    loader = test_loader if args.split == "test" else val_loader

    # Output
    plots_dir = Path(cfg["paths"]["plots"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_path = plots_dir / f"gradcam_{args.split}.png"

    visualize_gradcam(model, loader, device, save_path, n_samples=args.n)


if __name__ == "__main__":
    main()
