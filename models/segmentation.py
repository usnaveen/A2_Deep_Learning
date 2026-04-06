"""U-Net style semantic segmentation with VGG11 encoder.

Encoder: VGG11 convolutional blocks (5 blocks).
Decoder: Symmetric expansive path using ConvTranspose2d for upsampling
         and skip-connection concatenation at each stage.

Output: pixel-wise segmentation map with `num_classes` channels (3 for trimap).
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class DecoderBlock(nn.Module):
    """Single decoder block: upsample → concatenate skip → conv × 2."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # Learnable upsampling via transposed convolution
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        # After concat: in_channels + skip_channels → out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                         # upsample
        # Handle size mismatch from odd dimensions
        if x.shape != skip.shape:
            x = nn.functional.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)        # channel-wise concat
        x = self.conv(x)
        return x


class VGG11UNet(nn.Module):
    """U-Net style segmentation network with VGG11 encoder.

    Encoder (contracting path):
        Block 0: 3   → 64   (224 → 112)
        Block 1: 64  → 128  (112 → 56)
        Block 2: 128 → 256  (56  → 28)
        Block 3: 256 → 512  (28  → 14)
        Block 4: 512 → 512  (14  → 7)

    Bottleneck:
        512 → 1024 → 1024   (7 × 7)

    Decoder (expansive path):
        Up4: 1024 + 512(skip4) → 512  (7  → 14)
        Up3: 512  + 512(skip3) → 256  (14 → 28)
        Up2: 256  + 256(skip2) → 128  (28 → 56)
        Up1: 128  + 128(skip1) → 64   (56 → 112)
        Up0: 64   + 64 (skip0) → 64   (112→ 224)

    Final: 1×1 conv → num_classes
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output segmentation classes (3 for trimap).
            in_channels: Number of input channels.
            dropout_p: Dropout probability for bottleneck.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=True)

        # Bottleneck: 512 → 1024
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks (mirror the encoder)
        self.up4 = DecoderBlock(1024, 512, 512)   # 7  → 14, concat with block4 (512)
        self.up3 = DecoderBlock(512, 512, 256)    # 14 → 28, concat with block3 (512)
        self.up2 = DecoderBlock(256, 256, 128)    # 28 → 56, concat with block2 (256)
        self.up1 = DecoderBlock(128, 128, 64)     # 56 → 112, concat with block1 (128)
        self.up0 = DecoderBlock(64, 64, 64)       # 112→ 224, concat with block0 (64)

        # Final 1×1 convolution to produce class logits
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder with skip connections
        bottleneck_features, skips = self.encoder(x, return_features=True)
        # skips: block0 (64, 224), block1 (128, 112), block2 (256, 56),
        #        block3 (512, 28), block4 (512, 14)

        # Bottleneck
        x = self.bottleneck(bottleneck_features)   # [B, 1024, 7, 7]

        # Decoder path
        x = self.up4(x, skips["block4"])    # [B, 512, 14, 14]
        x = self.up3(x, skips["block3"])    # [B, 256, 28, 28]
        x = self.up2(x, skips["block2"])    # [B, 128, 56, 56]
        x = self.up1(x, skips["block1"])    # [B, 64, 112, 112]
        x = self.up0(x, skips["block0"])    # [B, 64, 224, 224]

        # Final classification
        logits = self.final_conv(x)          # [B, num_classes, 224, 224]
        return logits
