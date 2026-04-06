"""
DA6401 Assignment 2 — Google Colab Training Script
===================================================

Copy this to a Colab notebook cell-by-cell. 
Assumes the repo is cloned into /content/da6401_assignment_2.

RUNTIME → Change runtime type → T4 GPU (free tier)
"""

# %%
# ═══════════════════════════════════════════════════════
# CELL 1: Setup — Clone repo, install deps, login to W&B
# ═══════════════════════════════════════════════════════

# !git clone https://github.com/<YOUR_USERNAME>/da6401_assignment_2.git
# %cd da6401_assignment_2
# !pip install -q wandb albumentations gdown scikit-learn

import wandb
# wandb.login()  # Paste your API key when prompted

# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
# ═══════════════════════════════════════════════════════
# CELL 2: Download the Oxford-IIIT Pet Dataset
# ═══════════════════════════════════════════════════════

from data.pets_dataset import download_oxford_pet, create_dataloaders

download_oxford_pet("./data/oxford_pet")

# Quick sanity check
train_loader, val_loader, test_loader = create_dataloaders(
    root="./data/oxford_pet",
    batch_size=32,
    num_workers=2,
)
batch = next(iter(train_loader))
print(f"Image shape: {batch['image'].shape}")
print(f"Label shape: {batch['label'].shape}")
print(f"Bbox shape: {batch['bbox'].shape}")
print(f"Mask shape: {batch['mask'].shape}")

# %%
# ═══════════════════════════════════════════════════════
# CELL 3: §2.2 — DROPOUT EXPERIMENT (3 runs)
#   Run 1: No dropout (p=0)
#   Run 2: CustomDropout p=0.2
#   Run 3: CustomDropout p=0.5
# ═══════════════════════════════════════════════════════

import subprocess

for dropout in [0.0, 0.2, 0.5]:
    print(f"\n{'='*60}")
    print(f"Training classifier with dropout={dropout}")
    print(f"{'='*60}\n")
    subprocess.run([
        "python", "train.py",
        "--task", "classify",
        "--experiment", "exp-dropout",
        "--dropout", str(dropout),
        "--epochs", "30",
        "--lr", "1e-3",
        "--batch-size", "32",
        "--num-workers", "2",
    ])

# %%
# ═══════════════════════════════════════════════════════
# CELL 4: §2.1 — BATCHNORM EXPERIMENT (1 additional run)
#   Run without BatchNorm (the p=0.5 run above serves as "with BN")
# ═══════════════════════════════════════════════════════

subprocess.run([
    "python", "train.py",
    "--task", "classify",
    "--experiment", "exp-batchnorm",
    "--no-bn",
    "--dropout", "0.5",
    "--epochs", "30",
    "--lr", "1e-4",   # Lower LR since no BN
    "--batch-size", "32",
    "--num-workers", "2",
])

# %%
# ═══════════════════════════════════════════════════════
# CELL 5: §2.4 — FEATURE MAP VISUALIZATION (inference only)
# ═══════════════════════════════════════════════════════

subprocess.run([
    "python", "inference.py",
    "--mode", "featuremaps",
    "--num-workers", "2",
])

# %%
# ═══════════════════════════════════════════════════════
# CELL 6: TASK 2 — LOCALIZATION TRAINING
# ═══════════════════════════════════════════════════════

subprocess.run([
    "python", "train.py",
    "--task", "localize",
    "--experiment", "exp-detection",
    "--epochs", "30",
    "--lr", "1e-3",
    "--batch-size", "32",
    "--num-workers", "2",
])

# %%
# ═══════════════════════════════════════════════════════
# CELL 7: §2.3 — TRANSFER LEARNING SHOWDOWN (3 runs)
# ═══════════════════════════════════════════════════════

for freeze_mode in ["frozen", "partial", "full"]:
    print(f"\n{'='*60}")
    print(f"Training U-Net with freeze_mode={freeze_mode}")
    print(f"{'='*60}\n")
    subprocess.run([
        "python", "train.py",
        "--task", "segment",
        "--experiment", "exp-transfer",
        "--freeze-mode", freeze_mode,
        "--epochs", "30",
        "--lr", "1e-3" if freeze_mode != "frozen" else "1e-2",
        "--batch-size", "16",  # U-Net is memory hungry
        "--num-workers", "2",
    ])

# %%
# ═══════════════════════════════════════════════════════
# CELL 8: §2.7 — NOVEL IMAGE SHOWCASE
#   Download 3 pet images from the internet first
# ═══════════════════════════════════════════════════════

import urllib.request
import os

os.makedirs("novel_images", exist_ok=True)

# Example URLs — replace with actual pet images from the internet
novel_urls = [
    ("novel_images/dog1.jpg", "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=400"),
    ("novel_images/cat1.jpg", "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400"),
    ("novel_images/dog2.jpg", "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400"),
]

for path, url in novel_urls:
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        urllib.request.urlretrieve(url, path)

subprocess.run([
    "python", "inference.py",
    "--mode", "novel",
    "--images", "novel_images/dog1.jpg", "novel_images/cat1.jpg", "novel_images/dog2.jpg",
])

# %%
# ═══════════════════════════════════════════════════════
# CELL 9: Upload checkpoints to Google Drive
# ═══════════════════════════════════════════════════════

from google.colab import drive
drive.mount('/content/drive')

import shutil
drive_dir = "/content/drive/MyDrive/da6401_assignment2_checkpoints"
os.makedirs(drive_dir, exist_ok=True)

for ckpt in ["classifier.pth", "localizer.pth", "unet.pth"]:
    src = f"checkpoints/{ckpt}"
    if os.path.exists(src):
        shutil.copy(src, os.path.join(drive_dir, ckpt))
        print(f"✓ Copied {ckpt} to Google Drive")

print("\n✓ All checkpoints saved to Google Drive!")
print("Now go to Drive → Share each file → Get link → Note the file IDs")
print("Update models/multitask.py with the Drive IDs")
