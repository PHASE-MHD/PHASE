import math
from typing import Tuple, List, Optional, Union, Sequence, Any, Callable, Dict
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
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
from ..components.normalization import RMSNorm


class ReFiLMConditioner(nn.Module):
    """Small MLP that maps log-Re/log-ReM features to channel-wise FiLM."""

    def __init__(
        self,
        out_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        log_re_mean: float = 2.9756,
        log_re_std: float = 0.5417,
        include_rem: bool = True,
        residual_scale: float = 1.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.include_rem = bool(include_rem)
        self.residual_scale = float(residual_scale)
        self.register_buffer("log_re_mean", torch.tensor(float(log_re_mean)))
        self.register_buffer("log_re_std", torch.tensor(float(log_re_std)))

        in_dim = 2 if self.include_rem else 1
        layers = []
        current_dim = in_dim
        for _ in range(max(int(num_layers) - 1, 0)):
            layers.extend([nn.Linear(current_dim, int(hidden_dim)), nn.GELU()])
            current_dim = int(hidden_dim)
        layers.append(nn.Linear(current_dim, 2 * self.out_channels))
        self.net = nn.Sequential(*layers)

        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _normalize_re(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(1e-12)
        return (torch.log10(value) - self.log_re_mean) / self.log_re_std.clamp_min(1e-6)

    def _conditioning_features(self, re: torch.Tensor, rem: Optional[torch.Tensor]) -> torch.Tensor:
        if rem is None:
            rem = re
        features = [self._normalize_re(re).view(-1, 1)]
        if self.include_rem:
            features.append(self._normalize_re(rem).view(-1, 1))
        return torch.cat(features, dim=-1)

    def forward(self, re: torch.Tensor, rem: Optional[torch.Tensor] = None):
        gamma, beta = self.net(self._conditioning_features(re, rem)).chunk(2, dim=-1)
        return gamma, beta


class ReResidualAdapter2d(nn.Module):
    """Zero-initialized residual adapter for 2D UNet feature maps."""

    def __init__(
        self,
        dim: int,
        bottleneck_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        log_re_mean: float = 2.9756,
        log_re_std: float = 0.5417,
        include_rem: bool = True,
        adapter_scale: float = 1.0,
        gate_type: str = "channel",
    ):
        super().__init__()
        if gate_type not in {"scalar", "channel"}:
            raise ValueError("gate_type must be 'scalar' or 'channel'.")
        self.dim = int(dim)
        self.adapter_scale = float(adapter_scale)
        self.gate_type = gate_type
        self.conditioner = ReFiLMConditioner(
            out_channels=(self.dim if gate_type == "channel" else 1),
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            log_re_mean=log_re_mean,
            log_re_std=log_re_std,
            include_rem=include_rem,
            zero_init=True,
        )
        bottleneck_dim = int(min(max(1, bottleneck_dim), self.dim))
        self.norm = nn.GroupNorm(1, self.dim)
        self.down = nn.Conv2d(self.dim, bottleneck_dim, 1)
        self.act = nn.GELU()
        self.up = nn.Conv2d(bottleneck_dim, self.dim, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor, re: torch.Tensor, rem: Optional[torch.Tensor] = None) -> torch.Tensor:
        gamma, _beta = self.conditioner(re, rem)
        gate = 1.0 + gamma.to(hidden_states.dtype).view(hidden_states.shape[0], -1, 1, 1)
        update = self.up(self.act(self.down(self.norm(hidden_states))))
        return hidden_states + self.adapter_scale * gate * update


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
        re_conditioning: Optional[Dict[str, Any]] = None,
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

        self.re_conditioning_config = re_conditioning or {}
        self.re_conditioning_enabled = bool(self.re_conditioning_config.get("enabled", False))
        self.re_conditioning_type = self.re_conditioning_config.get("type", "output_film")
        self.re_reference = float(self.re_conditioning_config.get("reference_re", 1000.0))
        self.output_re_conditioning_enabled = (
            self.re_conditioning_enabled
            and self.re_conditioning_type in {"output_film", "deep_adapter_output_film"}
        )
        self.deep_re_conditioning_enabled = (
            self.re_conditioning_enabled
            and self.re_conditioning_type in {"deep_adapter", "deep_adapter_output_film"}
        )

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

        re_hidden_dim = self.re_conditioning_config.get("hidden_dim", 128)
        re_num_layers = self.re_conditioning_config.get("num_layers", 2)
        log_re_mean = self.re_conditioning_config.get("log_re_mean", 2.9756)
        log_re_std = self.re_conditioning_config.get("log_re_std", 0.5417)
        include_rem = self.re_conditioning_config.get("include_rem", True)
        residual_scale = self.re_conditioning_config.get("residual_scale", 1.0)
        deep_config = self.re_conditioning_config.get("deep_adapter", {})

        self.re_output_conditioner = None
        if self.output_re_conditioning_enabled:
            self.re_output_conditioner = ReFiLMConditioner(
                out_channels=self.out_dim,
                hidden_dim=re_hidden_dim,
                num_layers=re_num_layers,
                log_re_mean=log_re_mean,
                log_re_std=log_re_std,
                include_rem=include_rem,
                residual_scale=residual_scale,
                zero_init=True,
            )

        def make_adapter(adapter_dim):
            if not self.deep_re_conditioning_enabled:
                return nn.Identity()
            return ReResidualAdapter2d(
                dim=adapter_dim,
                bottleneck_dim=deep_config.get("bottleneck_dim", 64),
                hidden_dim=deep_config.get("hidden_dim", re_hidden_dim),
                num_layers=deep_config.get("num_layers", re_num_layers),
                log_re_mean=log_re_mean,
                log_re_std=log_re_std,
                include_rem=include_rem,
                adapter_scale=deep_config.get("adapter_scale", 1.0),
                gate_type=deep_config.get("gate_type", "channel"),
            )

        self.down_re_adapters = nn.ModuleList([
            nn.ModuleList([make_adapter(dim_in), make_adapter(dim_in)])
            for dim_in, _dim_out in in_out
        ])
        self.mid_re_adapters = nn.ModuleList([make_adapter(mid_dim), make_adapter(mid_dim)])
        self.up_re_adapters = nn.ModuleList([
            nn.ModuleList([make_adapter(dim_out), make_adapter(dim_out)])
            for _dim_in, dim_out in reversed(in_out)
        ])
        self.final_re_adapter = make_adapter(init_dim)

    def _prepare_re_tensor(
        self, value: Optional[torch.Tensor], batch: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if value is None:
            return torch.full((batch,), self.re_reference, device=device, dtype=dtype)
        value = value.to(device=device, dtype=dtype).view(-1)
        if value.numel() == 1 and batch != 1:
            value = value.expand(batch)
        if value.shape[0] != batch:
            raise ValueError(
                f"Re conditioning batch size mismatch: got {value.shape[0]} values for batch {batch}."
            )
        return value

    def _apply_re_adapter(self, adapter: nn.Module, x: torch.Tensor, re: Optional[torch.Tensor], rem: Optional[torch.Tensor]):
        if not self.deep_re_conditioning_enabled or isinstance(adapter, nn.Identity):
            return x
        re_tensor = self._prepare_re_tensor(re, x.shape[0], x.device, x.dtype)
        rem_tensor = self._prepare_re_tensor(rem, x.shape[0], x.device, x.dtype) if rem is not None else re_tensor
        return adapter(x, re_tensor, rem_tensor)

    def _apply_output_re_film(self, x: torch.Tensor, re: Optional[torch.Tensor], rem: Optional[torch.Tensor]):
        if self.re_output_conditioner is None:
            return x
        re_tensor = self._prepare_re_tensor(re, x.shape[0], x.device, x.dtype)
        rem_tensor = self._prepare_re_tensor(rem, x.shape[0], x.device, x.dtype) if rem is not None else re_tensor
        gamma, beta = self.re_output_conditioner(re_tensor, rem_tensor)
        gamma = gamma.to(x.dtype).view(x.shape[0], -1, 1, 1)
        beta = beta.to(x.dtype).view(x.shape[0], -1, 1, 1)
        return x + self.re_output_conditioner.residual_scale * (x * gamma + beta)

    def get_optimizer_parameters(self, optimizer_params):
        param_group_config = optimizer_params.get("param_groups", {})
        if not param_group_config or not param_group_config.get("enabled", False):
            return self.parameters()

        base_lr = param_group_config.get("pretrained_lr", optimizer_params.get("lr", 1e-4))
        new_lr = param_group_config.get("new_lr", optimizer_params.get("lr", 1e-4))
        weight_decay = optimizer_params.get("weight_decay", 0.0)
        new_weight_decay = param_group_config.get("new_weight_decay", weight_decay)
        base_weight_decay = param_group_config.get("pretrained_weight_decay", weight_decay)

        base_params = []
        re_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(("re_output_conditioner", "down_re_adapters", "mid_re_adapters", "up_re_adapters", "final_re_adapter")):
                re_params.append(param)
            else:
                base_params.append(param)

        groups = []
        if base_params:
            groups.append({"params": base_params, "lr": base_lr, "weight_decay": base_weight_decay, "name": "base_diffusion"})
        if re_params:
            groups.append({"params": re_params, "lr": new_lr, "weight_decay": new_weight_decay, "name": "re_conditioning"})
        return groups

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
        re: Optional[torch.Tensor] = None,
        rem: Optional[torch.Tensor] = None,
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

        for (block1, block2, attn, downsample), adapters in zip(self.downs, self.down_re_adapters):
            x = block1(x, t)
            x = self._apply_re_adapter(adapters[0], x, re, rem)
            h.append(x)

            x = block2(x, t)
            x = self._apply_re_adapter(adapters[1], x, re, rem)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self._apply_re_adapter(self.mid_re_adapters[0], x, re, rem)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)
        x = self._apply_re_adapter(self.mid_re_adapters[1], x, re, rem)

        for (block1, block2, attn, upsample), adapters in zip(self.ups, self.up_re_adapters):
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = self._apply_re_adapter(adapters[0], x, re, rem)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = self._apply_re_adapter(adapters[1], x, re, rem)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self._apply_re_adapter(self.final_re_adapter, x, re, rem)
        x = self.final_conv(x)
        return self._apply_output_re_film(x, re, rem)
