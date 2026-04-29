import torch
from pathlib import Path

# Path to your latest best model
model_path = Path("outputs/checkpoints/20260421_022556/best_model.pth")

if model_path.exists():
    checkpoint = torch.load(model_path, map_location="cpu")
    print(f"✅ Model found!")
    # In my updated code, I saved the epoch number in the checkpoint
    print(f"📅 This model was saved at Epoch: {checkpoint.get('epoch', 'Not found in ckpt')}")
else:
    print(f"❌ Could not find the file at {model_path}")