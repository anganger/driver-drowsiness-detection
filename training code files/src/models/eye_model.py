"""
eye_model.py
────────────
MobileNetV2 (and EfficientNet-B0) adapted for binary eye-state classification.

Design decisions:
  1. Pretrained ImageNet weights → strong low-level feature detectors (edges,
     textures) that transfer well to eye images.
  2. Two-phase fine-tuning:
       Phase 1 (warmup_epochs): freeze backbone, train only the new head.
                                 Prevents destroying pretrained weights with
                                 large early gradients.
       Phase 2 (remaining):     unfreeze all layers, use a lower LR to gently
                                 adapt the backbone.
  3. Custom head: GlobalAvgPool → Dropout → Linear(num_classes).
     Dropout(0.3) regularizes the head to prevent overfitting on the easy
     "open/closed" signal.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights,
)


# ──────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ──────────────────────────────────────────────────────────────────
class EyeStateClassifier(nn.Module):
    """
    Wraps a pretrained backbone with a custom classification head.

    Args:
        architecture : "mobilenetv2" or "efficientnet_b0"
        pretrained   : Use ImageNet weights if True
        num_classes  : 2  (open / closed)
        dropout      : Dropout rate before final linear layer
    """

    def __init__(
        self,
        architecture: str  = "mobilenetv2",
        pretrained:   bool = True,
        num_classes:  int  = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.architecture = architecture
        self.num_classes  = num_classes

        # ── Load backbone ──────────────────────────────────────
        if architecture == "mobilenetv2":
            weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
            base    = models.mobilenet_v2(weights=weights)
            in_features = base.classifier[1].in_features  # 1280

            # Replace original classifier
            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.backbone = base

        elif architecture == "efficientnet_b0":
            weights  = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base     = models.efficientnet_b0(weights=weights)
            in_features = base.classifier[1].in_features  # 1280

            base.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(in_features, num_classes),
            )
            self.backbone = base

        else:
            raise ValueError(
                f"Unknown architecture: {architecture}. "
                "Choose 'mobilenetv2' or 'efficientnet_b0'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, H, W]  (normalized)
        Returns logits: [B, num_classes]
        """
        return self.backbone(x)

    # ── Fine-tuning helpers ────────────────────────────────────
    def freeze_backbone(self):
        """
        Phase 1: Freeze all backbone parameters.
        Only the final classification head will be updated.
        """
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  🔒  Backbone frozen. Trainable params: "
              f"{trainable:,} / {total:,}  "
              f"({100*trainable/total:.1f}%)")

    def unfreeze_all(self):
        """
        Phase 2: Unfreeze all layers for full fine-tuning.
        Use a much lower LR to avoid catastrophic forgetting.
        """
        for param in self.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🔓  All layers unfrozen. Trainable params: {trainable:,}")

    def get_classifier_params(self) -> list:
        """Return only head parameters (for Phase 1 optimizer)."""
        return [p for n, p in self.named_parameters() if "classifier" in n]

    def get_backbone_params(self) -> list:
        """Return only backbone parameters (for Phase 2 lower LR)."""
        return [p for n, p in self.named_parameters() if "classifier" not in n]

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = self.num_trainable_params()
        print(f"\n  Model: {self.architecture}")
        print(f"  Total params    : {total:,}")
        print(f"  Trainable params: {trainable:,}")
        print(f"  Frozen params   : {total - trainable:,}")


# ──────────────────────────────────────────────────────────────────
# LOSS FUNCTION (with class-weight support)
# ──────────────────────────────────────────────────────────────────
def build_criterion(
    class_counts: torch.Tensor,
    device: torch.device,
    use_weights: bool = True,
) -> nn.CrossEntropyLoss:
    """
    CrossEntropyLoss with optional inverse-frequency class weights.

    Why: Even after WeightedRandomSampler, explicit class weights in the
    loss function provides an additional safety net and helps the gradient
    signal remain balanced.

    weight[c] = total / (num_classes * count[c])
    """
    if use_weights:
        total   = class_counts.float().sum()
        weights = total / (len(class_counts) * class_counts.float())
        weights = weights.to(device)
        print(f"  Loss weights: closed={weights[0]:.4f}, open={weights[1]:.4f}")
    else:
        weights = None

    return nn.CrossEntropyLoss(weight=weights)


# ──────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n=== Testing EyeStateClassifier ===\n")

    for arch in ["mobilenetv2", "efficientnet_b0"]:
        model = EyeStateClassifier(architecture=arch, pretrained=False)
        model.freeze_backbone()
        model.summary()

        dummy = torch.randn(4, 3, 224, 224)
        out   = model(dummy)
        print(f"  Output shape : {out.shape}")   # [4, 2]
        assert out.shape == (4, 2), "Shape mismatch!"
        print(f"  ✅  {arch} OK\n")

    print("All model tests passed!")
