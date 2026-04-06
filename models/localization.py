"""Localization module — VGG11-based single-object bounding box regression.

Uses a VGG11 encoder backbone with a regression head that outputs
[x_center, y_center, width, height] in pixel coordinates.
Trained with MSE + IoU loss.
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based localizer for single-object bounding box regression."""

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5, image_size: int = 224):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
            image_size: Input image spatial dimension (used for output scaling).
        """
        super().__init__()
        self.image_size = image_size
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=True)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),         # [x_center, y_center, w, h]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format in original image pixel space (not normalized values).
        """
        features = self.encoder(x)            # [B, 512, 7, 7]
        features = self.avgpool(features)     # [B, 512, 7, 7]
        features = torch.flatten(features, 1) # [B, 25088]
        bbox = self.head(features)            # [B, 4] — raw regression
        # Apply sigmoid and scale to image size so outputs are in valid pixel range
        bbox = torch.sigmoid(bbox) * self.image_size
        return bbox
