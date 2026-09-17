import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Callable


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization layer.

    This normalization technique normalizes by the root mean square,
    which can lead to more stable training in some cases compared to
    alternatives like LayerNorm or BatchNorm.
    """

    def __init__(self, dim: int):
        """
        Initialize RMS normalization.

        Args:
            dim: Feature dimension to normalize over
        """
        super().__init__()
        self.scale = dim**0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Normalized tensor with same shape
        """
        return F.normalize(x, dim=1) * self.g * self.scale
