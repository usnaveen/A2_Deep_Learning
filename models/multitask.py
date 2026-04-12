"""Unified multi-task model — shared backbone with three heads.

Loads pre-trained weights for the classifier, localizer, and U-Net,
then exposes a single forward() that returns all three predictions.

HOW TO UPDATE WEIGHTS FOR GRADESCOPE:
  1. Train each model on Kaggle (see Kaggle_Task*.ipynb notebooks).
  2. Upload the saved .pth files to Google Drive.
  3. Share each file as "Anyone with the link".
  4. Paste the file ID (from the share URL) into the three constants below.
  5. git commit + push — Gradescope will pick up the new IDs automatically.
"""

import os
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

# ── Paste your Google Drive file IDs here after uploading checkpoints ────────
# Get the ID from the share link:  https://drive.google.com/file/d/FILE_ID/view
CLASSIFIER_DRIVE_ID = "1Ob4ecaMOaoH-56TdIQ9lYEdGKfn1jy8L"   # ← classifier (val F1=0.9325)
LOCALIZER_DRIVE_ID  = "1jRM83Xr6HETheUhv1MQqRMEl2eAwbkSK"   # ← localizer (val Acc@0.5=91.8%)
UNET_DRIVE_ID       = "1527bugKX9NKd_JAOltXdH_rpU1j6Spzj"   # ← partial freeze unet (val Dice≈0.82)


def _canonicalize_checkpoint(sd: dict) -> dict:
    """Normalize checkpoint key names to our layer naming convention.

    Handles checkpoints saved with a nested block layout (blockN.M.L) and
    alternate head names (classifier / regressor), converting everything to
    the flat convention used by this codebase (blockX.Y, head).
    Returns the dict unchanged if it already matches our format.
    """
    import re
    # Detect nested format by looking for 3-level encoder keys
    if not any(re.match(r'encoder\.block\d+\.\d+\.\d+\.', k) for k in sd):
        return sd  # already our format — nothing to do

    new_sd: dict = {}
    for k, v in sd.items():
        m = re.match(r'encoder\.block(\d+)\.(\d+)\.(\d+)\.(.*)', k)
        if m:
            N, M, L, suf = int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)
            X = N - 1                       # 1-indexed → 0-indexed
            if X in (0, 1):                 # single-conv block
                Y = 0 if L == 0 else 1
            else:                           # double-conv block
                Y = (0 if L == 0 else 1) if M == 0 else (3 if L == 0 else 4)
            new_sd[f'encoder.block{X}.{Y}.{suf}'] = v
        elif k.startswith('classifier.'):
            new_sd['head.' + k[len('classifier.'):]] = v
        elif k.startswith('regressor.'):
            new_sd['head.' + k[len('regressor.'):]] = v
        # keys not matching our encoder layout (e.g. pool layers) are dropped
    return new_sd


def _safe_download(drive_id: str, output_path: str, label: str) -> bool:
    """Download a file from Google Drive only if it doesn't exist locally.

    Returns True if the file is available (already existed or just downloaded).
    """
    if os.path.exists(output_path):
        print(f"  ✓ {label} already present at {output_path}")
        return True

    if drive_id.startswith("<"):
        print(f"  ⚠  {label}: Drive ID not set yet — skipping download.")
        return False

    try:
        import gdown
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"  ↓ Downloading {label} from Google Drive ...")
        gdown.download(id=drive_id, output=output_path, quiet=False, fuzzy=True)
        if os.path.exists(output_path):
            print(f"  ✓ {label} downloaded successfully.")
            return True
        else:
            print(f"  ✗ {label}: download completed but file not found.")
            return False
    except Exception as exc:
        print(f"  ✗ {label}: download failed — {exc}")
        return False


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    A single forward pass yields:
      - classification logits  [B, num_breeds]
      - bounding box coords    [B, 4]   (xcenter, ycenter, w, h  in pixels)
      - segmentation logits    [B, seg_classes, H, W]
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "checkpoints/classifier.pth",
        localizer_path: str = "checkpoints/localizer.pth",
        unet_path: str = "checkpoints/unet.pth",
        image_size: int = 224,
    ):
        """
        Initialize the shared backbone/heads using trained weights.

        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained unet weights.
            image_size: Expected input spatial size for bbox scaling.
        """
        super().__init__()

        # ── Download checkpoints from Drive if not already present ───────
        _safe_download(CLASSIFIER_DRIVE_ID, classifier_path, "classifier.pth")
        _safe_download(LOCALIZER_DRIVE_ID,  localizer_path,  "localizer.pth")
        _safe_download(UNET_DRIVE_ID,       unet_path,       "unet.pth")

        self.image_size = image_size

        # ── Shared VGG11 encoder ─────────────────────────────────────────
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=True)

        # ── Classification head (from classifier) ────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classification_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, num_breeds),
        )

        # ── Localization head (regression → bbox) ────────────────────────
        self.localization_head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Linear(1024, 4),
        )

        # ── Segmentation decoder (U-Net style) ──────────────────────────
        self.seg_bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=0.5),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        from .segmentation import DecoderBlock
        self.seg_up4 = DecoderBlock(1024, 512, 512)
        self.seg_up3 = DecoderBlock(512,  512, 256)
        self.seg_up2 = DecoderBlock(256,  256, 128)
        self.seg_up1 = DecoderBlock(128,  128,  64)
        self.seg_up0 = DecoderBlock( 64,   64,  64)
        self.seg_final = nn.Conv2d(64, seg_classes, kernel_size=1)

        # ── Load pretrained weights ─────────────────────────────────────
        self._load_pretrained(classifier_path, localizer_path, unet_path)

    def _load_pretrained(self, cls_path: str, loc_path: str, unet_path: str):
        """Load weights from individually trained models."""
        device = "cpu"

        # Load classifier weights → encoder + classification_head
        if os.path.exists(cls_path):
            cls_state = _canonicalize_checkpoint(
                torch.load(cls_path, map_location=device, weights_only=False)
            )
            # Load encoder weights from classifier
            encoder_keys = {k: v for k, v in cls_state.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(
                {k.replace("encoder.", ""): v for k, v in encoder_keys.items()},
                strict=False,
            )
            # Load classification head
            head_keys = {k: v for k, v in cls_state.items() if k.startswith("head.")}
            if head_keys:
                self.classification_head.load_state_dict(
                    {k.replace("head.", ""): v for k, v in head_keys.items()},
                    strict=False,
                )
            print("  ✓ Classifier weights loaded.")

        # Load localizer weights → localization_head
        if os.path.exists(loc_path):
            loc_state = _canonicalize_checkpoint(
                torch.load(loc_path, map_location=device, weights_only=False)
            )
            head_keys = {k: v for k, v in loc_state.items() if k.startswith("head.")}
            if head_keys:
                self.localization_head.load_state_dict(
                    {k.replace("head.", ""): v for k, v in head_keys.items()},
                    strict=False,
                )
            print("  ✓ Localizer weights loaded.")

        # Load U-Net weights → segmentation decoder
        if os.path.exists(unet_path):
            unet_state = torch.load(unet_path, map_location=device, weights_only=False)
            # Load bottleneck
            bn_keys = {k.replace("bottleneck.", ""): v for k, v in unet_state.items()
                       if k.startswith("bottleneck.")}
            if bn_keys:
                self.seg_bottleneck.load_state_dict(bn_keys, strict=False)
            # Load decoder blocks
            for block_name in ["up4", "up3", "up2", "up1", "up0"]:
                src_prefix = f"{block_name}."
                dst_block = getattr(self, f"seg_{block_name}")
                block_keys = {k.replace(src_prefix, ""): v for k, v in unet_state.items()
                              if k.startswith(src_prefix)}
                if block_keys:
                    dst_block.load_state_dict(block_keys, strict=False)
            # Final conv
            fc_keys = {k.replace("final_conv.", ""): v for k, v in unet_state.items()
                       if k.startswith("final_conv.")}
            if fc_keys:
                self.seg_final.load_state_dict(fc_keys, strict=False)
            print("  ✓ U-Net segmentation weights loaded.")

    def forward(self, x: torch.Tensor):
        """Forward pass for multi-task model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            A dict with keys:
            - 'classification': [B, num_breeds] logits tensor.
            - 'localization': [B, 4] bounding box tensor.
            - 'segmentation': [B, seg_classes, H, W] segmentation logits tensor.
        """
        # ── Shared encoder with skip connections ─────────────────────────
        bottleneck, skips = self.encoder(x, return_features=True)

        # ── Classification branch ────────────────────────────────────────
        cls_features = self.avgpool(bottleneck)
        cls_features = torch.flatten(cls_features, 1)
        classification = self.classification_head(cls_features)

        # ── Localization branch ──────────────────────────────────────────
        loc_features = self.avgpool(bottleneck)
        loc_features = torch.flatten(loc_features, 1)
        localization = self.localization_head(loc_features)
        localization = torch.sigmoid(localization) * self.image_size

        # ── Segmentation branch (U-Net decoder) ─────────────────────────
        seg = self.seg_bottleneck(bottleneck)
        seg = self.seg_up4(seg, skips["block4"])
        seg = self.seg_up3(seg, skips["block3"])
        seg = self.seg_up2(seg, skips["block2"])
        seg = self.seg_up1(seg, skips["block1"])
        seg = self.seg_up0(seg, skips["block0"])
        segmentation = self.seg_final(seg)

        return {
            "classification": classification,
            "localization":   localization,
            "segmentation":   segmentation,
        }
