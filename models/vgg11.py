"""VGG11 encoder — Configuration A from Simonyan & Zisserman (2014).

Layer configuration: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
where numbers = out_channels and 'M' = MaxPool2d(2, 2).

BatchNorm2d is inserted after every Conv2d (before ReLU) following the
VGG-BN variant convention.
"""

from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


# VGG11 configuration A — (out_channels | 'M' for maxpool)
VGG11_CFG: List[Union[int, str]] = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with optional intermediate feature returns.

    The encoder is split into 5 blocks separated by MaxPool so that
    skip-connection features can be extracted for U-Net style decoders.

    Block 0: Conv(3  -> 64)  + BN + ReLU  →  pool  (features: 64  x 112 x 112)
    Block 1: Conv(64 -> 128) + BN + ReLU  →  pool  (features: 128 x 56  x 56 )
    Block 2: Conv(128-> 256) * 2 + BN     →  pool  (features: 256 x 28  x 28 )
    Block 3: Conv(256-> 512) * 2 + BN     →  pool  (features: 512 x 14  x 14 )
    Block 4: Conv(512-> 512) * 2 + BN     →  pool  (features: 512 x 7   x 7  )
    """

    def __init__(self, in_channels: int = 3, use_bn: bool = True):
        """Initialize the VGG11Encoder model.

        Args:
            in_channels: Number of input image channels (default 3 for RGB).
            use_bn: Whether to include BatchNorm2d layers after each Conv2d.
        """
        super().__init__()
        self.use_bn = use_bn

        # Build the 5 encoder blocks (conv layers within each block, pool at end)
        self.block0, ch = self._make_block(in_channels, [64], use_bn)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1, ch = self._make_block(ch, [128], use_bn)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block2, ch = self._make_block(ch, [256, 256], use_bn)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block3, ch = self._make_block(ch, [512, 512], use_bn)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block4, ch = self._make_block(ch, [512, 512], use_bn)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.out_channels = ch  # 512

    @staticmethod
    def _make_block(in_ch: int, channels: List[int], use_bn: bool) -> Tuple[nn.Sequential, int]:
        """Build a VGG conv block (conv + bn + relu) * N."""
        layers: List[nn.Module] = []
        for out_ch in channels:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        return nn.Sequential(*layers), in_ch

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W] (expects 224×224).
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B, 512, 7, 7].
            - if return_features=True: (bottleneck, feature_dict) where
              feature_dict maps block names to pre-pool feature tensors.
        """
        features: Dict[str, torch.Tensor] = {}

        # Block 0: 224 -> 112
        x0 = self.block0(x)
        if return_features:
            features["block0"] = x0       # [B, 64, 224, 224] before pool
        x = self.pool0(x0)

        # Block 1: 112 -> 56
        x1 = self.block1(x)
        if return_features:
            features["block1"] = x1       # [B, 128, 112, 112]
        x = self.pool1(x1)

        # Block 2: 56 -> 28
        x2 = self.block2(x)
        if return_features:
            features["block2"] = x2       # [B, 256, 56, 56]
        x = self.pool2(x2)

        # Block 3: 28 -> 14
        x3 = self.block3(x)
        if return_features:
            features["block3"] = x3       # [B, 512, 28, 28]
        x = self.pool3(x3)

        # Block 4: 14 -> 7
        x4 = self.block4(x)
        if return_features:
            features["block4"] = x4       # [B, 512, 14, 14]
        x = self.pool4(x4)

        if return_features:
            return x, features
        return x


class VGG11(nn.Module):
    """Full VGG11 network with classification head.

    This is the class the autograder imports as `from models.vgg11 import VGG11`.
    It wraps VGG11Encoder + AdaptiveAvgPool + FC classifier.
    """

    def __init__(
        self,
        num_classes: int = 37,
        in_channels: int = 3,
        dropout_p: float = 0.5,
        use_bn: bool = True,
    ):
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels, use_bn=use_bn)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x