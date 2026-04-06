"""Inference and evaluation — W&B report experiments.

Usage:
  # §2.7: Novel image pipeline showcase
  python inference.py --mode novel --images img1.jpg img2.jpg img3.jpg

  # §2.5: Re-run detection table logging
  python inference.py --mode detection

  # §2.4: Feature map visualization
  python inference.py --mode featuremaps
"""

import argparse
import os
from pathlib import Path
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import wandb

from data.pets_dataset import (
    OxfordIIITPetDataset, get_val_transforms,
    IMAGENET_MEAN, IMAGENET_STD, create_dataloaders,
)
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from train import denormalize_image, compute_iou, compute_dice_score, compute_pixel_accuracy


def load_models(checkpoint_dir: str, device: torch.device, image_size: int = 224):
    """Load all three trained models."""
    cls_model = VGG11Classifier(num_classes=37).to(device)
    loc_model = VGG11Localizer(image_size=image_size).to(device)
    seg_model = VGG11UNet(num_classes=3).to(device)

    cls_path = Path(checkpoint_dir) / "classifier.pth"
    loc_path = Path(checkpoint_dir) / "localizer.pth"
    unet_path = Path(checkpoint_dir) / "unet.pth"

    if cls_path.exists():
        cls_model.load_state_dict(torch.load(str(cls_path), map_location=device, weights_only=False))
    if loc_path.exists():
        loc_model.load_state_dict(torch.load(str(loc_path), map_location=device, weights_only=False))
    if unet_path.exists():
        seg_model.load_state_dict(torch.load(str(unet_path), map_location=device, weights_only=False))

    cls_model.eval()
    loc_model.eval()
    seg_model.eval()
    return cls_model, loc_model, seg_model


def preprocess_image(image_path: str, image_size: int = 224):
    """Load and preprocess a single image for inference."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
    transformed = transform(image=img_np)
    return transformed["image"].unsqueeze(0), img_np  # [1, 3, H, W], original


def novel_image_showcase(args):
    """§2.7: Run full pipeline on 3 novel pet images from the internet."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    run = wandb.init(
        project=args.wandb_project,
        name="novel_images_showcase",
        tags=["exp-novel", "inference"],
        config={"mode": "novel_images"},
    )

    cls_model, loc_model, seg_model = load_models(args.checkpoint_dir, device, args.image_size)

    # Trimap colormap
    colormap = np.array([[0, 200, 0], [40, 40, 40], [255, 255, 0]], dtype=np.uint8)
    class_names = OxfordIIITPetDataset.CLASS_NAMES

    results = []

    for img_path in args.images:
        img_tensor, orig_img = preprocess_image(img_path, args.image_size)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            # Classification
            cls_logits = cls_model(img_tensor)
            cls_probs = torch.softmax(cls_logits, dim=1)
            cls_conf, cls_pred = cls_probs.max(dim=1)
            breed = class_names[cls_pred.item()]
            confidence = cls_conf.item()

            # Localization
            bbox = loc_model(img_tensor).cpu().numpy()[0]  # [cx, cy, w, h]

            # Segmentation
            seg_logits = seg_model(img_tensor)
            seg_mask = seg_logits.argmax(dim=1).cpu().numpy()[0]

        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Original + breed label
        resized = np.array(Image.fromarray(orig_img).resize((args.image_size, args.image_size)))
        axes[0].imshow(resized)
        axes[0].set_title(f"Breed: {breed}\nConfidence: {confidence:.3f}", fontsize=11)
        axes[0].axis("off")

        # Panel 2: Bounding box overlay
        img_denorm = denormalize_image(img_tensor[0])
        axes[1].imshow(img_denorm)
        rect = patches.Rectangle(
            (bbox[0] - bbox[2]/2, bbox[1] - bbox[3]/2), bbox[2], bbox[3],
            linewidth=3, edgecolor='red', facecolor='none',
        )
        axes[1].add_patch(rect)
        axes[1].set_title("Bounding Box Prediction", fontsize=11)
        axes[1].axis("off")

        # Panel 3: Segmentation mask
        mask_colored = colormap[seg_mask]
        axes[2].imshow(mask_colored)
        axes[2].set_title("Segmentation Mask", fontsize=11)
        axes[2].axis("off")

        plt.suptitle(f"Pipeline Output: {os.path.basename(img_path)}", fontsize=13, fontweight='bold')
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        results.append(wandb.Image(buf, caption=f"{breed} ({confidence:.2f}) — {os.path.basename(img_path)}"))

    wandb.log({"novel_images/pipeline_output": results})
    wandb.finish()
    print("✓ Novel image showcase logged to W&B")


def feature_map_visualization(args):
    """§2.4: Visualize feature maps from first and last conv layers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    run = wandb.init(
        project=args.wandb_project,
        name="feature_map_visualization",
        tags=["exp-featuremaps", "inference"],
        config={"mode": "featuremaps"},
    )

    cls_model = VGG11Classifier(num_classes=37).to(device)
    cls_path = Path(args.checkpoint_dir) / "classifier.pth"
    if cls_path.exists():
        cls_model.load_state_dict(torch.load(str(cls_path), map_location=device, weights_only=False))
    cls_model.eval()

    # Register hooks
    activations = {}
    def make_hook(name):
        def hook_fn(module, input, output):
            activations[name] = output.detach().cpu()
        return hook_fn

    # First conv layer (block0, first Conv2d)
    cls_model.encoder.block0[0].register_forward_hook(make_hook("first_conv"))
    # Last conv layer (block4, last Conv2d)
    last_conv_idx = 0
    for i, layer in enumerate(cls_model.encoder.block4):
        if isinstance(layer, nn.Conv2d):
            last_conv_idx = i
    cls_model.encoder.block4[last_conv_idx].register_forward_hook(make_hook("last_conv"))

    # Get a test image (use a dog image)
    _, val_loader, _ = create_dataloaders(
        root=args.data_root, image_size=args.image_size,
        batch_size=1, num_workers=0,
    )
    batch = next(iter(val_loader))
    img = batch["image"].to(device)

    with torch.no_grad():
        _ = cls_model(img)

    # Visualize
    for layer_name in ["first_conv", "last_conv"]:
        feat = activations[layer_name][0]  # [C, H, W]
        n_channels = min(32, feat.shape[0])

        # Grid of feature maps
        n_cols = 8
        n_rows = (n_channels + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
        axes = axes.flatten()

        for c in range(n_channels):
            fm = feat[c].numpy()
            fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
            axes[c].imshow(fm, cmap="viridis")
            axes[c].set_title(f"Ch {c}", fontsize=7)
            axes[c].axis("off")

        for c in range(n_channels, len(axes)):
            axes[c].axis("off")

        plt.suptitle(f"Feature Maps: {layer_name}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)

        wandb.log({f"feature_maps/{layer_name}": wandb.Image(buf, caption=layer_name)})

    # Also log original image
    img_np = denormalize_image(img[0])
    wandb.log({"feature_maps/input_image": wandb.Image(img_np, caption="Input Image")})

    wandb.finish()
    print("✓ Feature map visualization logged to W&B")


def parse_args():
    parser = argparse.ArgumentParser(description="DA6401 — Inference & Evaluation")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["novel", "detection", "featuremaps"],
                        help="Which inference mode to run")
    parser.add_argument("--images", nargs="+", default=[],
                        help="Image paths for novel image showcase")
    parser.add_argument("--wandb-project", type=str, default="da6401-assignment2")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
    parser.add_argument("--data-root", type=str, default="./data/oxford_pet")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "novel":
        if not args.images:
            print("Error: --images required for novel mode")
            return
        novel_image_showcase(args)
    elif args.mode == "featuremaps":
        feature_map_visualization(args)
    elif args.mode == "detection":
        # Re-run detection table logging
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        run = wandb.init(
            project=args.wandb_project,
            name="detection_table",
            tags=["exp-detection", "inference"],
        )
        _, _, test_loader = create_dataloaders(
            root=args.data_root, image_size=args.image_size,
            batch_size=16, num_workers=4,
        )
        loc_model = VGG11Localizer(image_size=args.image_size).to(device)
        loc_path = Path(args.checkpoint_dir) / "localizer.pth"
        if loc_path.exists():
            loc_model.load_state_dict(torch.load(str(loc_path), map_location=device, weights_only=False))
        loc_model.eval()
        from train import log_detection_table
        log_detection_table(loc_model, test_loader, device, args.image_size)
        wandb.finish()


if __name__ == "__main__":
    main()