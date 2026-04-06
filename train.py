"""Training entrypoint — unified script for all tasks and W&B experiments.

Usage examples:
  # Task 1: Classification
  python train.py --task classify --experiment exp-dropout --dropout 0.5 --epochs 30

  # Task 1: BatchNorm ablation
  python train.py --task classify --experiment exp-batchnorm --no-bn --epochs 30

  # Task 2: Localization
  python train.py --task localize --experiment exp-detection --epochs 30

  # Task 3: Segmentation — transfer learning experiments
  python train.py --task segment --experiment exp-transfer --freeze-mode frozen --epochs 30
  python train.py --task segment --experiment exp-transfer --freeze-mode partial --epochs 30
  python train.py --task segment --experiment exp-transfer --freeze-mode full --epochs 30

All runs are logged to W&B with the appropriate experiment tags for the report.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

import wandb

# Local imports
from data.pets_dataset import create_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from models.classification import VGG11Classifier
from models.localization import VGG11Localizer
from models.segmentation import VGG11UNet
from losses.iou_loss import IoULoss


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_dice_score(pred: torch.Tensor, target: torch.Tensor, num_classes: int = 3, eps: float = 1e-6):
    """Compute per-class Dice score and return the mean."""
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    dice_per_class = []
    for c in range(num_classes):
        pred_c = (pred_flat == c).float()
        target_c = (target_flat == c).float()
        intersection = (pred_c * target_c).sum()
        dice = (2.0 * intersection + eps) / (pred_c.sum() + target_c.sum() + eps)
        dice_per_class.append(dice.item())
    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor):
    """Pixel-level accuracy."""
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total


def compute_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor, eps: float = 1e-6):
    """Compute IoU between predicted and target boxes (cxcywh format). Returns per-sample IoU."""
    # Convert to xyxy
    px1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    py1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    px2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    py2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

    tx1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
    ty1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tx2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
    ty2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

    inter_x1 = torch.max(px1, tx1)
    inter_y1 = torch.max(py1, ty1)
    inter_x2 = torch.min(px2, tx2)
    inter_y2 = torch.min(py2, ty2)

    inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    target_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
    union_area = pred_area + target_area - inter_area

    return inter_area / (union_area + eps)


# ──────────────────────────────────────────────────────────────────────────────
# Activation hook for §2.1 (BatchNorm experiment)
# ──────────────────────────────────────────────────────────────────────────────

class ActivationLogger:
    """Register hooks to log activation distributions from specific layers."""

    def __init__(self):
        self.activations = {}
        self._hooks = []

    def register(self, model: nn.Module, layer_name: str, layer: nn.Module):
        def hook_fn(module, input, output, name=layer_name):
            self.activations[name] = output.detach().cpu()
        self._hooks.append(layer.register_forward_hook(hook_fn))

    def clear(self):
        self.activations = {}

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ──────────────────────────────────────────────────────────────────────────────
# Training loops
# ──────────────────────────────────────────────────────────────────────────────

def train_classifier(args):
    """Train the VGG11 classifier (Task 1).

    W&B experiments served:
      - exp-batchnorm (§2.1): with/without BN, log 3rd conv activation distributions
      - exp-dropout   (§2.2): p=0 / 0.2 / 0.5, log train vs val loss
      - exp-featuremaps (§2.4): after training, log feature maps from 1st/last conv
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    use_bn = not args.no_bn

    # Init W&B
    run = wandb.init(
        project=args.wandb_project,
        name=f"classify_bn={use_bn}_drop={args.dropout}",
        tags=[args.experiment, "classification"],
        config={
            "task": "classification",
            "use_bn": use_bn,
            "dropout_p": args.dropout,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "image_size": args.image_size,
        },
    )

    # Data
    train_loader, val_loader, test_loader = create_dataloaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Model
    model = VGG11Classifier(
        num_classes=37,
        dropout_p=args.dropout,
        use_bn=use_bn,
    ).to(device)

    # Register activation hooks for §2.1 (3rd conv layer)
    act_logger = ActivationLogger()
    # 3rd conv layer is in block2 (first conv of block2 — the 3rd conv overall)
    if hasattr(model.encoder, "block2"):
        act_logger.register(model, "conv3_block2", model.encoder.block2[0])  # First Conv2d in block2
    # Also hook first and last conv for §2.4 feature maps
    if hasattr(model.encoder, "block0"):
        act_logger.register(model, "conv1_block0", model.encoder.block0[0])
    if hasattr(model.encoder, "block4"):
        # Last Conv2d in block4
        last_conv_idx = 0
        for i, layer in enumerate(model.encoder.block4):
            if isinstance(layer, nn.Conv2d):
                last_conv_idx = i
        act_logger.register(model, "conv_last_block4", model.encoder.block4[last_conv_idx])

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        epoch_start = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total
        epoch_time = time.time() - epoch_start

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                logits = model(images)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        # ── Log to W&B ──
        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/f1_macro": val_f1,
            "epoch_time_s": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        }

        # §2.1: Log activation distribution of 3rd conv every 5 epochs
        if "conv3_block2" in act_logger.activations and epoch % 5 == 0:
            act = act_logger.activations["conv3_block2"]
            log_dict["activations/conv3_histogram"] = wandb.Histogram(act.flatten().numpy())
            log_dict["activations/conv3_mean"] = act.mean().item()
            log_dict["activations/conv3_std"] = act.std().item()

        wandb.log(log_dict, step=epoch)
        act_logger.clear()

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | Time: {epoch_time:.1f}s")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = Path(args.checkpoint_dir) / "classifier.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(save_path))
            print(f"  ✓ Saved best classifier (F1={val_f1:.4f})")

    # ── §2.4: Feature map visualization after training ──
    model.eval()
    with torch.no_grad():
        # Get a single image
        sample = next(iter(val_loader))
        single_img = sample["image"][:1].to(device)
        _ = model(single_img)  # trigger hooks

        # Log feature maps
        for layer_name in ["conv1_block0", "conv_last_block4"]:
            if layer_name in act_logger.activations:
                feat = act_logger.activations[layer_name][0]  # [C, H, W]
                # Log first 16 channels as images
                images_list = []
                for c in range(min(16, feat.shape[0])):
                    fm = feat[c].numpy()
                    fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                    images_list.append(wandb.Image(fm, caption=f"{layer_name}_ch{c}"))
                wandb.log({f"feature_maps/{layer_name}": images_list})

    act_logger.remove_hooks()
    wandb.finish()
    print(f"\n✓ Classification training complete. Best val F1: {best_val_f1:.4f}")


def train_localizer(args):
    """Train the VGG11 localizer (Task 2).

    W&B experiments served:
      - exp-detection (§2.5): log detection table with images, GT/pred boxes, IoU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    run = wandb.init(
        project=args.wandb_project,
        name=f"localize_lr={args.lr}",
        tags=[args.experiment, "localization"],
        config={
            "task": "localization",
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "image_size": args.image_size,
        },
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Optionally load pretrained encoder from classifier
    model = VGG11Localizer(dropout_p=args.dropout, image_size=args.image_size).to(device)
    cls_ckpt = Path(args.checkpoint_dir) / "classifier.pth"
    if cls_ckpt.exists():
        print(f"Loading pretrained encoder from {cls_ckpt}")
        cls_state = torch.load(str(cls_ckpt), map_location=device, weights_only=False)
        encoder_state = {k.replace("encoder.", ""): v for k, v in cls_state.items()
                         if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_state, strict=False)

    # Loss: MSE + IoU (as specified in README)
    mse_loss = nn.MSELoss()
    iou_loss = IoULoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_iou = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_iou_sum = 0.0
        train_total = 0
        epoch_start = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            bboxes = batch["bbox"].to(device)

            optimizer.zero_grad()
            pred_boxes = model(images)
            loss_mse = mse_loss(pred_boxes, bboxes)
            loss_iou = iou_loss(pred_boxes, bboxes)
            loss = loss_mse + loss_iou
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ious = compute_iou(pred_boxes, bboxes)
                train_iou_sum += ious.sum().item()

            train_loss_sum += loss.item() * images.size(0)
            train_total += images.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_total
        train_iou = train_iou_sum / train_total
        epoch_time = time.time() - epoch_start

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_iou_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                bboxes = batch["bbox"].to(device)
                pred_boxes = model(images)
                loss_mse = mse_loss(pred_boxes, bboxes)
                loss_iou = iou_loss(pred_boxes, bboxes)
                loss = loss_mse + loss_iou
                val_loss_sum += loss.item() * images.size(0)
                ious = compute_iou(pred_boxes, bboxes)
                val_iou_sum += ious.sum().item()
                val_total += images.size(0)

        val_loss = val_loss_sum / val_total
        val_iou = val_iou_sum / val_total

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "train/iou": train_iou,
            "val/loss": val_loss,
            "val/iou": val_iou,
            "epoch_time_s": epoch_time,
        }, step=epoch)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f} IoU: {val_iou:.4f} | Time: {epoch_time:.1f}s")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_path = Path(args.checkpoint_dir) / "localizer.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(save_path))
            print(f"  ✓ Saved best localizer (IoU={val_iou:.4f})")

    # ── §2.5: Log detection table with bounding box overlays ──
    log_detection_table(model, test_loader, device, args.image_size)

    wandb.finish()
    print(f"\n✓ Localization training complete. Best val IoU: {best_val_iou:.4f}")


def train_segmentation(args):
    """Train the VGG11 U-Net segmentation model (Task 3).

    W&B experiments served:
      - exp-transfer (§2.3): frozen / partial / full fine-tuning comparison
      - exp-segmentation (§2.6): Dice vs Pixel Accuracy tracking + sample masks
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    freeze_mode = args.freeze_mode  # "frozen", "partial", "full"

    run = wandb.init(
        project=args.wandb_project,
        name=f"segment_{freeze_mode}_lr={args.lr}",
        tags=[args.experiment, "segmentation", f"freeze-{freeze_mode}"],
        config={
            "task": "segmentation",
            "freeze_mode": freeze_mode,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "image_size": args.image_size,
        },
    )

    train_loader, val_loader, test_loader = create_dataloaders(
        root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = VGG11UNet(num_classes=3, dropout_p=args.dropout).to(device)

    # Load pretrained encoder weights from classifier
    cls_ckpt = Path(args.checkpoint_dir) / "classifier.pth"
    if cls_ckpt.exists():
        print(f"Loading pretrained encoder from {cls_ckpt}")
        cls_state = torch.load(str(cls_ckpt), map_location=device, weights_only=False)
        encoder_state = {k.replace("encoder.", ""): v for k, v in cls_state.items()
                         if k.startswith("encoder.")}
        model.encoder.load_state_dict(encoder_state, strict=False)

    # ── Apply freeze strategy for §2.3 ──
    if freeze_mode == "frozen":
        # Freeze entire encoder
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Strategy: FROZEN — entire encoder frozen")

    elif freeze_mode == "partial":
        # Freeze early blocks (0, 1, 2), unfreeze later blocks (3, 4)
        for name, param in model.encoder.named_parameters():
            if any(name.startswith(b) for b in ["block0", "block1", "block2",
                                                  "pool0", "pool1", "pool2"]):
                param.requires_grad = False
            else:
                param.requires_grad = True
        print("Strategy: PARTIAL — blocks 0-2 frozen, blocks 3-4 trainable")

    elif freeze_mode == "full":
        # All parameters trainable
        for param in model.encoder.parameters():
            param.requires_grad = True
        print("Strategy: FULL — entire network trainable")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} parameters")
    wandb.config.update({"trainable_params": trainable_params, "total_params": total_params})

    # Loss: weighted cross-entropy (class imbalance in trimaps)
    # Class weights: boundary is rare, background is common
    class_weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_dice = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_dice_sum = 0.0
        train_pixacc_sum = 0.0
        train_total = 0
        epoch_start = time.time()

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            logits = model(images)          # [B, 3, H, W]
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_masks = logits.argmax(dim=1)
                dice = compute_dice_score(pred_masks, masks)
                pixacc = compute_pixel_accuracy(pred_masks, masks)

            train_loss_sum += loss.item() * images.size(0)
            train_dice_sum += dice * images.size(0)
            train_pixacc_sum += pixacc * images.size(0)
            train_total += images.size(0)

        scheduler.step()
        train_loss = train_loss_sum / train_total
        train_dice = train_dice_sum / train_total
        train_pixacc = train_pixacc_sum / train_total
        epoch_time = time.time() - epoch_start

        # Validate
        model.eval()
        val_loss_sum = 0.0
        val_dice_sum = 0.0
        val_pixacc_sum = 0.0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                pred_masks = logits.argmax(dim=1)
                dice = compute_dice_score(pred_masks, masks)
                pixacc = compute_pixel_accuracy(pred_masks, masks)
                val_loss_sum += loss.item() * images.size(0)
                val_dice_sum += dice * images.size(0)
                val_pixacc_sum += pixacc * images.size(0)
                val_total += images.size(0)

        val_loss = val_loss_sum / val_total
        val_dice = val_dice_sum / val_total
        val_pixacc = val_pixacc_sum / val_total

        log_dict = {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/dice": train_dice,
            "train/pixel_accuracy": train_pixacc,
            "val/loss": val_loss,
            "val/dice": val_dice,
            "val/pixel_accuracy": val_pixacc,
            "epoch_time_s": epoch_time,
        }

        # §2.6: Log sample segmentation masks every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            log_segmentation_samples(model, val_loader, device, epoch)

        wandb.log(log_dict, step=epoch)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} Dice: {val_dice:.4f} PixAcc: {val_pixacc:.4f} | Time: {epoch_time:.1f}s")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            save_path = Path(args.checkpoint_dir) / "unet.pth"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(save_path))
            print(f"  ✓ Saved best U-Net (Dice={val_dice:.4f})")

    wandb.finish()
    print(f"\n✓ Segmentation training complete. Best val Dice: {best_val_dice:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# W&B logging helpers
# ──────────────────────────────────────────────────────────────────────────────

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize a [3, H, W] tensor back to [0, 255] uint8 numpy array."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def log_detection_table(model, test_loader, device, image_size):
    """§2.5: Log W&B table with detection results — GT boxes green, predictions red."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from io import BytesIO

    model.eval()
    columns = ["Image", "GT Box", "Pred Box", "IoU", "Confidence"]
    table = wandb.Table(columns=columns)

    count = 0
    max_samples = 15

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            gt_boxes = batch["bbox"]            # [B, 4] cxcywh
            pred_boxes = model(images).cpu()    # [B, 4] cxcywh

            ious = compute_iou(pred_boxes, gt_boxes)

            for i in range(images.size(0)):
                if count >= max_samples:
                    break

                img_np = denormalize_image(images[i])
                gt = gt_boxes[i].numpy()
                pred = pred_boxes[i].numpy()
                iou_val = ious[i].item()

                # Draw boxes on image
                fig, ax = plt.subplots(1, figsize=(4, 4))
                ax.imshow(img_np)

                # GT box (green) — convert cxcywh to xywh for Rectangle
                gt_rect = patches.Rectangle(
                    (gt[0] - gt[2]/2, gt[1] - gt[3]/2), gt[2], gt[3],
                    linewidth=2, edgecolor='green', facecolor='none', label='GT'
                )
                ax.add_patch(gt_rect)

                # Pred box (red)
                pred_rect = patches.Rectangle(
                    (pred[0] - pred[2]/2, pred[1] - pred[3]/2), pred[2], pred[3],
                    linewidth=2, edgecolor='red', facecolor='none', label='Pred'
                )
                ax.add_patch(pred_rect)

                ax.set_title(f"IoU: {iou_val:.3f}")
                ax.legend(fontsize=8)
                ax.axis("off")
                plt.tight_layout()

                # Save to buffer
                buf = BytesIO()
                fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)

                # Confidence = 1 / (1 + MSE) as a proxy
                mse_val = ((pred - gt) ** 2).mean()
                confidence = 1.0 / (1.0 + mse_val)

                table.add_data(
                    wandb.Image(buf),
                    f"cx={gt[0]:.1f} cy={gt[1]:.1f} w={gt[2]:.1f} h={gt[3]:.1f}",
                    f"cx={pred[0]:.1f} cy={pred[1]:.1f} w={pred[2]:.1f} h={pred[3]:.1f}",
                    round(iou_val, 4),
                    round(float(confidence), 4),
                )
                count += 1

            if count >= max_samples:
                break

    wandb.log({"detection_results": table})


def log_segmentation_samples(model, val_loader, device, epoch, n_samples=5):
    """§2.6: Log segmentation samples — original, GT mask, predicted mask."""
    model.eval()
    images_list = []
    count = 0

    # Color map for trimap: foreground=green, background=black, boundary=yellow
    colormap = np.array([[0, 200, 0], [40, 40, 40], [255, 255, 0]], dtype=np.uint8)

    with torch.no_grad():
        for batch in val_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"]
            logits = model(imgs)
            pred_masks = logits.argmax(dim=1).cpu()

            for i in range(imgs.size(0)):
                if count >= n_samples:
                    break

                img_np = denormalize_image(imgs[i])
                gt_colored = colormap[masks[i].numpy()]
                pred_colored = colormap[pred_masks[i].numpy()]

                images_list.append(wandb.Image(img_np, caption=f"Original (ep{epoch})"))
                images_list.append(wandb.Image(gt_colored, caption=f"GT Mask (ep{epoch})"))
                images_list.append(wandb.Image(pred_colored, caption=f"Pred Mask (ep{epoch})"))
                count += 1

            if count >= n_samples:
                break

    wandb.log({f"segmentation_samples/epoch_{epoch}": images_list})


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 — Training")

    # Task selection
    parser.add_argument("--task", type=str, required=True,
                        choices=["classify", "localize", "segment"],
                        help="Which task to train")

    # W&B experiment
    parser.add_argument("--experiment", type=str, default="default",
                        help="W&B experiment tag (e.g. exp-dropout, exp-batchnorm, exp-transfer)")
    parser.add_argument("--wandb-project", type=str, default="da6401-assignment2",
                        help="W&B project name")

    # Hyperparameters
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")

    # Task-specific
    parser.add_argument("--no-bn", action="store_true", help="Disable BatchNorm (for §2.1)")
    parser.add_argument("--freeze-mode", type=str, default="full",
                        choices=["frozen", "partial", "full"],
                        help="Transfer learning strategy for segmentation (§2.3)")

    # Data & I/O
    parser.add_argument("--data-root", type=str, default="./data/oxford_pet",
                        help="Root dir for dataset")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


def main():
    args = parse_args()

    print(f"╔══════════════════════════════════════════╗")
    print(f"║  DA6401 Assignment 2 — Task: {args.task:>10s}  ║")
    print(f"║  Experiment: {args.experiment:>27s}  ║")
    print(f"╚══════════════════════════════════════════╝")

    if args.task == "classify":
        train_classifier(args)
    elif args.task == "localize":
        train_localizer(args)
    elif args.task == "segment":
        train_segmentation(args)


if __name__ == "__main__":
    main()