"""Classification components — VGG11Classifier.

Wraps VGG11Encoder + a 3-layer FC classifier head with CustomDropout.
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Classifier(nn.Module):
    """Full classifier = VGG11Encoder + ClassificationHead."""

    def __init__(self, num_classes: int = 37, in_channels: int = 3, dropout_p: float = 0.5, use_bn: bool = True):
        """
        Initialize the VGG11Classifier model.

        Args:
            num_classes: Number of output classes.
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the classifier head.
            use_bn: Whether to use BatchNorm in the encoder.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=use_bn)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.head = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for classification model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Classification logits [B, num_classes].
        """
        features = self.encoder(x)          # [B, 512, 7, 7]
        features = self.avgpool(features)   # [B, 512, 7, 7]
        features = torch.flatten(features, 1)  # [B, 512*7*7]
        logits = self.head(features)        # [B, num_classes]
        return logits
