"""Standard normalization that scales data using mean and standard deviation."""

import torch
import torch.nn as nn

from .normalization_factory import register_normalization


class StandardNormalization(nn.Module):
    """
    Standard normalization that scales data using mean and standard deviation.

    This normalization scales the data to have zero mean and unit variance
    based on the provided statistics.

    Args:
        mean: Pre-computed mean values per channel
        std: Pre-computed standard deviation values per channel
        eps: Small value to avoid division by zero
    """

    def __init__(self, mean=None, std=None, eps=1e-8):
        super().__init__()
        self.eps = eps

        # Register mean and std as buffers
        if mean is not None:
            self.register_buffer("mean", torch.tensor(mean))
        else:
            self.mean = None

        if std is not None:
            self.register_buffer("std", torch.tensor(std))
        else:
            self.std = None

        # # Debug prints
        # print(f"mean: {self.mean}")
        # print(f"std: {self.std}")

    def normalize(self, x):
        """Apply normalization to input"""
        if self.mean is None or self.std is None:
            return x

        mean = self.mean.to(x.device).view(1, -1, 1, 1, 1)
        std = self.std.to(x.device).view(1, -1, 1, 1, 1)

        return (x - mean) / (std + self.eps)

    def denormalize(self, x):
        """Apply denormalization to output"""
        if self.mean is None or self.std is None:
            return x

        mean = self.mean.to(x.device).view(1, -1, 1, 1, 1)
        std = self.std.to(x.device).view(1, -1, 1, 1, 1)

        return x * (std + self.eps) + mean

    def to(self, device):
        """Override to method to properly handle device transfers"""
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        return self


@register_normalization("standard")
def create_standard_normalization(norm_params):
    """Create a standard normalization."""
    return StandardNormalization(
        mean=norm_params.get("mean"),
        std=norm_params.get("std"),
        eps=norm_params.get("eps", 1e-8),
    )
