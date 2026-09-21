import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import wraps, partial
from packaging import version
from collections import namedtuple
from typing import Optional, List, Tuple, Union, Callable
from einops import rearrange, reduce, repeat

from ...utils import exists, default, once, print_once
from .normalization import RMSNorm


AttentionConfig = namedtuple(
    "AttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"]
)


class Attend(nn.Module):
    """
    Core attention mechanism implementation with various backend options.

    This module provides a flexible attention implementation that can leverage
    different backend optimizations including flash attention when available.
    It automatically selects the most efficient implementation based on the
    available hardware and PyTorch version.
    """

    def __init__(
        self, dropout: float = 0.0, flash: bool = False, scale: Optional[float] = None
    ):
        """
        Initialize the attention module.

        Args:
            dropout: Dropout probability for attention weights
            flash: Whether to use flash attention when available
            scale: Optional custom scaling factor for attention (default: 1/sqrt(dim))
        """
        super().__init__()
        self.dropout = dropout
        self.scale = scale
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        self.flash_available = False

        if flash and version.parse(torch.__version__) >= version.parse("2.0.0"):
            self.flash_available = True
        else:
            if flash:
                print_once(
                    "Flash attention requested but PyTorch version < 2.0.0, falling back to regular attention\n"
                )
            self.flash_available = False

        self.cpu_config = AttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not self.flash_available:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        device_version = version.parse(
            f"{device_properties.major}.{device_properties.minor}"
        )

        if device_version >= version.parse("8.0"):
            print_once("\nA100/H100 GPU detected (compute capability >= 8.0)\n")
            # Enable all attention mechanisms as fallback options
            self.cuda_config = AttentionConfig(True, True, True)
        else:
            print_once(
                "\nNon-A100/H100 GPU detected, using math or mem efficient attention if input tensor is on cuda\n"
            )
            self.cuda_config = AttentionConfig(False, True, True)

    def flash_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention using PyTorch's optimized SDPA (Scaled Dot-Product Attention).

        Args:
            q: Query tensor [batch, heads, seq_len, dim]
            k: Key tensor [batch, heads, seq_len, dim]
            v: Value tensor [batch, heads, seq_len, dim]

        Returns:
            Output tensor after attention [batch, heads, seq_len, dim]
        """
        _, heads, q_len, _, k_len, is_cuda, device = (
            *q.shape,
            k.shape[-2],
            q.is_cuda,
            q.device,
        )

        if exists(self.scale):
            default_scale = q.shape[-1]
            q = q * (self.scale / default_scale)

        q, k, v = map(lambda t: t.contiguous(), (q, k, v))

        config = self.cuda_config if is_cuda else self.cpu_config

        dropout_p = self.dropout if self.training else 0.0

        # PyTorch 2.1+ exposes torch.nn.attention.sdpa_kernel; older 2.x builds
        # use torch.backends.cuda.sdp_kernel. Support both so older containers
        # can select the requested backend without crashing during evaluation.
        if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
            backends = []
            if config.enable_flash:
                backends.append(torch.nn.attention.SDPBackend.FLASH_ATTENTION)
            if config.enable_math:
                backends.append(torch.nn.attention.SDPBackend.MATH)
            if config.enable_mem_efficient:
                backends.append(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION)

            if backends:
                with torch.nn.attention.sdpa_kernel(backends):
                    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
            else:
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        elif is_cuda and hasattr(torch.backends.cuda, "sdp_kernel"):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=config.enable_flash,
                enable_math=config.enable_math,
                enable_mem_efficient=config.enable_mem_efficient,
            ):
                out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        return out

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the attention module.

        Implements the standard query-key-value attention mechanism with automatic
        selection of the most efficient backend.

        Args:
            q: Query tensor [batch, heads, seq_len, dim]
            k: Key tensor [batch, heads, seq_len, dim]
            v: Value tensor [batch, heads, seq_len, dim]

        Returns:
            Output tensor after attention [batch, heads, seq_len, dim]
        """
        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        if self.flash and self.flash_available:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)
        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out


class Attention(nn.Module):
    """
    Multi-head self-attention module with learned memory vectors.

    This module implements multi-head self-attention with additional
    learned memory key-value pairs, suitable for processing image features
    in a convolutional neural network.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        dim_head: int = 32,
        num_mem_kv: int = 4,
        flash: bool = False,
    ):
        """
        Initialize the attention module.

        Args:
            dim: Input feature dimension
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            num_mem_kv: Number of memory key-value pairs
            flash: Whether to use flash attention if available
        """
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention with memory key-values.

        Args:
            x: Input feature map [batch, channels, height, width]

        Returns:
            Attention output with same shape as input
        """
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h n d -> b h n d", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Linear attention implementation for more efficient computation.

    This module implements a linearized version of attention using the
    associative property of matrix multiplication, resulting in O(n)
    complexity instead of O(n²) for standard attention, making it more
    efficient for longer sequences.
    """

    def __init__(
        self, dim: int, heads: int = 4, dim_head: int = 32, num_mem_kv: int = 4
    ):
        """
        Initialize the linear attention module.

        Args:
            dim: Input feature dimension
            heads: Number of attention heads
            dim_head: Dimension of each attention head
            num_mem_kv: Number of memory key-value pairs
        """
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply linear attention to the input.

        Args:
            x: Input feature map [batch, channels, height, width]

        Returns:
            Attention output with same shape as input
        """
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        mk, mv = map(lambda t: repeat(t, "h c n -> b h c n", b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
