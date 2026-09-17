"""Identity normalization that doesn't modify the data."""

import torch
import torch.nn as nn

from .normalization_factory import register_normalization


class IdentityNormalization(nn.Module):
    """
    Identity normalization that preserves original values.

    This is a passthrough that doesn't modify the input or output data.
    """

    def __init__(self, **kwargs):
        super().__init__()

    def normalize(self, x):
        """Apply normalization to input"""
        return x

    def denormalize(self, x):
        """Apply denormalization to output"""
        return x


@register_normalization("identity")
def create_identity_normalization(norm_params):
    """Create a simple identity normalization that does nothing."""
    return IdentityNormalization()
