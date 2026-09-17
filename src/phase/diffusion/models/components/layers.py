import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Callable, Any
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
import math

from ...utils import exists, default, cast_tuple, divisible_by
from .blocks import ResnetBlock
from .attention import Attention, LinearAttention
from .normalization import RMSNorm


def Upsample(
    dim: int,
    dim_out: Optional[int] = None,
    padding_mode: str = "zeros",
) -> nn.Sequential:
    """
    Create an upsampling module with 2x scaling.

    Args:
        dim: Input dimension
        dim_out: Output dimension (defaults to dim if not specified)

    Returns:
        Sequential module that upsamples spatial dimensions by a factor of 2
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1, padding_mode=padding_mode),
    )


def Downsample(dim: int, dim_out: Optional[int] = None) -> nn.Sequential:
    """
    Create a downsampling module with 2x scaling.

    Uses pixel rearrangement for efficient downsampling followed by a 1x1 convolution.

    Args:
        dim: Input dimension
        dim_out: Output dimension (defaults to dim if not specified)

    Returns:
        Sequential module that downsamples spatial dimensions by a factor of 2
    """
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embedding module.

    This module creates position embeddings using sine and cosine functions
    of different frequencies, following the approach in "Attention Is All You Need".
    It's used to encode time steps in diffusion models.
    """

    def __init__(self, dim: int, theta: int = 10000):
        """
        Initialize sinusoidal position embedding.

        Args:
            dim: Embedding dimension (must be even)
            theta: Frequency scaling parameter
        """
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute sinusoidal embeddings for input values.

        Args:
            x: Input tensor of arbitrary shape

        Returns:
            Sinusoidal embeddings with shape (*x.shape, dim)
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """
    Random or learned sinusoidal position embedding.

    Following @crowsonkb's implementation, this module provides either random
    or learned sinusoidal positional embeddings, which can be more flexible
    than fixed embeddings.

    References:
        https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8
    """

    def __init__(self, dim: int, is_random: bool = False):
        """
        Initialize random or learned sinusoidal embedding.

        Args:
            dim: Embedding dimension (must be even)
            is_random: If True, use random non-trainable weights; if False, use learned weights
        """
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute embeddings for input values.

        Args:
            x: Input tensor of arbitrary shape

        Returns:
            Embeddings with shape (*x.shape, dim+1) including the original input
        """
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
