import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Callable
from einops import rearrange, reduce, repeat
from functools import partial

from ...utils import exists, default
from .attention import Attention, LinearAttention
from .normalization import RMSNorm


class Block(nn.Module):
    """
    Basic convolutional block with normalization and activation.

    This block consists of a 2D convolution, normalization layer,
    optional scale-shift conditioning, activation, and dropout.
    It's used as a building block for more complex architectures.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        dropout: float = 0.0,
        padding_mode: str = "zeros",
    ):
        """
        Initialize the block.

        Args:
            dim: Input dimension
            dim_out: Output dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1, padding_mode=padding_mode)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        scale_shift: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the block.

        Args:
            x: Input tensor
            scale_shift: Optional tuple of (scale, shift) for feature-wise conditioning

        Returns:
            Processed feature map
        """
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(nn.Module):
    """
    Residual block with optional time embedding conditioning.

    This block implements a residual connection around two Block modules,
    with optional conditioning from a time embedding. It's a core building
    block for diffusion model architectures.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        *,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        padding_mode: str = "zeros",
    ):
        """
        Initialize the ResNet block.

        Args:
            dim: Input dimension
            dim_out: Output dimension
            time_emb_dim: Optional time embedding dimension for conditioning
            dropout: Dropout probability
        """
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, dropout=dropout, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, padding_mode=padding_mode)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the ResNet block.

        Args:
            x: Input tensor
            time_emb: Optional time embedding for conditioning

        Returns:
            Output feature map with residual connection
        """
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)
