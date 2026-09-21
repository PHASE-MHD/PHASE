from typing import Tuple, Optional, Union, Sequence
import torch
from torch import nn
from functools import partial

from ...utils import exists, default, cast_tuple, divisible_by
from ..components.layers import (
    Upsample,
    Downsample,
    SinusoidalPosEmb,
    RandomOrLearnedSinusoidalPosEmb,
)
from ..components.blocks import ResnetBlock
from ..components.attention import Attention, LinearAttention


class UNet(nn.Module):
    """
    UNet architecture that serves as the backbone for diffusion models.

    This implementation is specifically designed to be used with diffusion models
    and provides a flexible architecture with skip connections, attention mechanisms,
    and time embeddings for conditioning. It can be configured with different dimensions,
    self-conditioning, and attention mechanisms.

    The architecture follows the standard U-Net design with encoder path (downs),
    bottleneck, and decoder path (ups) with skip connections between matching
    resolutions of encoder and decoder.
    """

    def __init__(
        self,
        dim: int,
        init_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 3,
        self_condition: bool = False,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10000,
        dropout: float = 0.0,
        attn_dim_head: Union[int, Sequence[int]] = 32,
        attn_heads: Union[int, Sequence[int]] = 4,
        full_attn: Optional[
            Union[bool, Tuple[bool, ...]]
        ] = None,  # defaults to full attention only for inner most layer
        flash_attn: bool = False,
        padding_mode: str = "zeros",
    ):
        """
        Initialize the UNet model.

        Args:
            dim: Base dimension for the model
            init_dim: Initial dimension after first convolution (default: same as dim)
            out_dim: Output dimension (default: channels × 2 if learned_variance else channels)
            dim_mults: Dimension multipliers for each level of the UNet
            channels: Number of input/output channels
            self_condition: Whether to use self-conditioning
            learned_variance: Whether to predict variance as well (outputs 2×channels)
            learned_sinusoidal_cond: Whether to learn sinusoidal embeddings
            random_fourier_features: Whether to use random Fourier features
            learned_sinusoidal_dim: Dimension of learned sinusoidal embeddings
            sinusoidal_pos_emb_theta: Parameter for sinusoidal position embeddings
            dropout: Dropout probability
            attn_dim_head: Dimension of each attention head
            attn_heads: Number of attention heads
            full_attn: Whether to use full attention or linear attention per level
            flash_attn: Whether to use flash attention for efficient computation
        """
        super().__init__()

        self.channels = channels
        self.self_condition = self_condition
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError(
                "padding_mode must be one of 'zeros', 'reflect', 'replicate', 'circular'."
            )
        self.padding_mode = padding_mode
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(
            input_channels, init_dim, 7, padding=3, padding_mode=self.padding_mode
        )

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # attention
        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(
            ResnetBlock,
            time_emb_dim=time_dim,
            dropout=dropout,
            padding_mode=self.padding_mode,
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_in, dim_in),
                        resnet_block(dim_in, dim_in),
                        attn_klass(
                            dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads
                        ),
                        (
                            Downsample(dim_in, dim_out)
                            if not is_last
                            else nn.Conv2d(
                                dim_in, dim_out, 3, padding=1, padding_mode=self.padding_mode
                            )
                        ),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(
            mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
        )
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(
                nn.ModuleList(
                    [
                        resnet_block(dim_out + dim_in, dim_out),
                        resnet_block(dim_out + dim_in, dim_out),
                        attn_klass(
                            dim_out,
                            dim_head=layer_attn_dim_head,
                            heads=layer_attn_heads,
                        ),
                        (
                            Upsample(dim_out, dim_in, padding_mode=self.padding_mode)
                            if not is_last
                            else nn.Conv2d(
                                dim_out, dim_in, 3, padding=1, padding_mode=self.padding_mode
                            )
                        ),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self) -> int:
        """
        Calculate the total downsampling factor of the UNet.

        Returns:
            The factor by which spatial dimensions are reduced in the encoder path
        """
        return 2 ** (len(self.downs) - 1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        x_self_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the UNet model.

        Args:
            x: Input image tensor [batch, channels, height, width]
            time: Time embedding tensor [batch, 1]
            x_self_cond: Optional self-conditioning input (same shape as x)

        Returns:
            Output tensor [batch, out_dim, height, width]

        Raises:
            AssertionError: If input dimensions aren't divisible by the downsampling factor
        """
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), (
            f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"
        )

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
