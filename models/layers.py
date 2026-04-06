"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer using inverted dropout.

    During training, randomly zeros elements with probability `p` and
    scales remaining elements by 1/(1-p) so that expected values are
    preserved.  During evaluation the input is returned unchanged.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability.
        """
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of shape [B, C, H, W] or [B, D].

        Returns:
            Output tensor with dropout applied (training) or identity (eval).
        """
        if not self.training or self.p == 0.0:
            return x

        # Binary mask: 1 = keep, 0 = drop
        mask = (torch.rand_like(x.float()) >= self.p).to(x.dtype)
        # Inverted dropout: scale by 1/(1-p) so E[output] == E[input]
        return x * mask / (1.0 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"
