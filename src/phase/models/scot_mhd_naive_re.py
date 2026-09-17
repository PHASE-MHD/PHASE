"""Naive multi-regime scOT conditioning through constant Re/Rm input maps."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .scot_mhd import PoseidonMHDFinetune


class PoseidonMHDReInputFinetune(PoseidonMHDFinetune):
    """Append standardized log10(Re) and log10(Rm) maps to the scOT input."""

    accepts_conditioning = True

    def __init__(
        self,
        log_re_mean: float = 2.9756,
        log_re_std: float = 0.5417,
        include_rem: bool = True,
        re_reference: float = 1000.0,
        re_channel_init: str = "zero",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if re_channel_init not in {"zero", "mean_velocity", "mean_all"}:
            raise ValueError(
                "re_channel_init must be 'zero', 'mean_velocity', or 'mean_all'."
            )

        self.include_rem = include_rem
        self.re_reference = float(re_reference)
        self.re_num_conditioning_channels = 2 if include_rem else 1
        self.register_buffer("log_re_mean", torch.tensor(float(log_re_mean)))
        self.register_buffer("log_re_std", torch.tensor(float(log_re_std)))
        self._re_input_hooks_registered = False
        self._expand_input_projection_for_re_channels(re_channel_init)

    def _expand_input_projection_for_re_channels(self, init: str) -> None:
        patch_embeddings = self.poseidon.embeddings.patch_embeddings
        projection = patch_embeddings.projection
        if not isinstance(projection, nn.Conv2d):
            raise TypeError(
                "Expected Poseidon patch embedding projection to be nn.Conv2d, "
                f"got {type(projection)!r}."
            )

        old_in_channels = int(projection.in_channels)
        self.warm_start_input_channels = old_in_channels
        new_projection = nn.Conv2d(
            in_channels=old_in_channels + self.re_num_conditioning_channels,
            out_channels=projection.out_channels,
            kernel_size=projection.kernel_size,
            stride=projection.stride,
            padding=projection.padding,
            dilation=projection.dilation,
            groups=projection.groups,
            bias=projection.bias is not None,
            padding_mode=projection.padding_mode,
        ).to(device=projection.weight.device, dtype=projection.weight.dtype)

        with torch.no_grad():
            new_projection.weight[:, :old_in_channels].copy_(projection.weight)
            if init == "zero":
                new_projection.weight[:, old_in_channels:].zero_()
            elif init == "mean_velocity":
                source = projection.weight[
                    :, list(self.poseidon_input_channel_map)
                ].mean(dim=1, keepdim=True)
                new_projection.weight[:, old_in_channels:].copy_(
                    source.expand(-1, self.re_num_conditioning_channels, -1, -1)
                )
            else:
                source = projection.weight.mean(dim=1, keepdim=True)
                new_projection.weight[:, old_in_channels:].copy_(
                    source.expand(-1, self.re_num_conditioning_channels, -1, -1)
                )
            if projection.bias is not None:
                new_projection.bias.copy_(projection.bias)

        patch_embeddings.projection = new_projection
        patch_embeddings.num_channels = new_projection.in_channels
        self.poseidon.config.num_channels = new_projection.in_channels
        self.poseidon_num_channels = new_projection.in_channels

    def _register_re_input_gradient_scaling(
        self, old_to_new_lr_ratio: float
    ) -> None:
        if self._re_input_hooks_registered:
            return
        ratio = float(old_to_new_lr_ratio)

        def scale_warm_started_input_channels(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:, : self.warm_start_input_channels] *= ratio
            return grad

        self.poseidon.embeddings.patch_embeddings.projection.weight.register_hook(
            scale_warm_started_input_channels
        )
        self._re_input_hooks_registered = True

    def get_optimizer_parameters(self, optimizer_params):
        """Assign the high learning rate only to the new Re/Rm input slice."""
        group_config = optimizer_params.get("param_groups", {})
        if not group_config or not group_config.get("enabled", False):
            return self.parameters()

        pretrained_lr = group_config.get(
            "pretrained_lr", optimizer_params.get("lr", 1e-5)
        )
        new_lr = group_config.get("new_lr", optimizer_params.get("lr", 1e-5))
        weight_decay = optimizer_params.get("weight_decay", 0.0)
        pretrained_weight_decay = group_config.get(
            "pretrained_weight_decay", weight_decay
        )
        new_weight_decay = group_config.get("new_weight_decay", weight_decay)

        re_input_name = "poseidon.embeddings.patch_embeddings.projection.weight"
        pretrained_params = []
        re_input_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name == re_input_name:
                re_input_params.append(parameter)
            else:
                pretrained_params.append(parameter)

        if re_input_params and new_lr != 0:
            self._register_re_input_gradient_scaling(pretrained_lr / new_lr)

        groups = []
        if pretrained_params:
            groups.append(
                {
                    "params": pretrained_params,
                    "lr": pretrained_lr,
                    "weight_decay": pretrained_weight_decay,
                    "name": "warm_started_mhd",
                }
            )
        if re_input_params:
            groups.append(
                {
                    "params": re_input_params,
                    "lr": new_lr,
                    "weight_decay": new_weight_decay,
                    "name": "naive_re_rm_input_channels",
                }
            )
        return groups

    def _normalize_re(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(1e-12)
        return (torch.log10(value) - self.log_re_mean) / self.log_re_std.clamp_min(
            1e-6
        )

    def _prepare_conditioning(
        self,
        value: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if value is None:
            value = torch.full((batch_size,), self.re_reference, device=device)
        else:
            value = value.to(device=device).view(-1)
            if value.numel() == 1 and batch_size != 1:
                value = value.expand(batch_size)
        if value.shape[0] != batch_size:
            raise ValueError(
                "Re input-channel conditioning batch size mismatch: "
                f"got {value.shape[0]} values for batch {batch_size}."
            )
        return self._normalize_re(value).to(dtype=dtype)

    def adapt_mhd_to_poseidon(
        self,
        fields: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        rem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        adapted = super().adapt_mhd_to_poseidon(fields)[
            :, : self.warm_start_input_channels
        ]
        batch_size, _, height, width = adapted.shape
        re_feature = self._prepare_conditioning(
            re, batch_size, adapted.device, adapted.dtype
        )
        if rem is None:
            rem = re
        rem_feature = self._prepare_conditioning(
            rem, batch_size, adapted.device, adapted.dtype
        )

        features = [re_feature]
        if self.include_rem:
            features.append(rem_feature)
        maps = [
            feature.view(batch_size, 1, 1, 1).expand(
                batch_size, 1, height, width
            )
            for feature in features
        ]
        conditioned = torch.cat([adapted, *maps], dim=1)
        expected = self.poseidon.embeddings.patch_embeddings.num_channels
        if conditioned.shape[1] != expected:
            raise RuntimeError(
                "Naive Re/Rm input construction produced "
                f"{conditioned.shape[1]} channels, but the Poseidon patch "
                f"embedding expects {expected}."
            )
        return conditioned

    def forward_transition(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        rem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pixel_values = self.adapt_mhd_to_poseidon(x, re=re, rem=rem)
        output = self.poseidon(pixel_values=pixel_values, time=time)
        return self.extract_mhd_prediction(output.output, initial_fields=x)

    def forward(
        self,
        x: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        re: Optional[torch.Tensor] = None,
        rem: Optional[torch.Tensor] = None,
    ):
        if x.dim() == 4:
            if time is None:
                time = x.new_ones(x.shape[0])
            return self.forward_transition(x, time, re=re, rem=rem)
        if x.dim() != 5:
            raise ValueError(
                "Expected x with shape [B, 3, H, W] or [B, C, T, H, W], "
                f"got {tuple(x.shape)}."
            )
        if x.shape[1] < 6:
            raise ValueError(
                "5D compatibility mode expects at least 6 channels: "
                "[t, x, y, u0, v0, A0]."
            )

        predictions = []
        for time_index in range(x.shape[2]):
            transition_input = x[:, 3:6, time_index]
            if time is None:
                step_time = x[:, 0, time_index, 0, 0]
            elif time.dim() == 2:
                step_time = time[:, time_index]
            else:
                step_time = time
            predictions.append(
                self.forward_transition(transition_input, step_time, re=re, rem=rem)
            )
        return torch.stack(predictions, dim=2)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Expand a five-channel single-Re checkpoint for seven-channel input."""
        key = "poseidon.embeddings.patch_embeddings.projection.weight"
        if key in state_dict:
            target = self.state_dict()[key]
            source = state_dict[key]
            if (
                source.ndim == target.ndim
                and source.shape[0] == target.shape[0]
                and source.shape[2:] == target.shape[2:]
                and source.shape[1] < target.shape[1]
            ):
                expanded = target.detach().clone()
                expanded[:, : source.shape[1]].copy_(source)
                state_dict = dict(state_dict)
                state_dict[key] = expanded
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def create_poseidon_mhd_re_input_finetune(params):
    """Build the locked naive Re/Rm input-conditioning model."""
    re_config = params.get("re_input_conditioning", {})
    return PoseidonMHDReInputFinetune(
        poseidon_model=params.get("poseidon_model", "camlab-ethz/Poseidon-T"),
        load_pretrained_poseidon=params.get("load_pretrained_poseidon", True),
        image_size=params.get("image_size", 128),
        patch_size=params.get("patch_size", 4),
        out_channels=params.get("out_channels", 3),
        poseidon_input_channel_map=tuple(
            params.get("poseidon_input_channel_map", [1, 2])
        ),
        poseidon_output_channel_map=tuple(
            params.get("poseidon_output_channel_map", [1, 2])
        ),
        magnetic_channel_index=params.get("magnetic_channel_index", 4),
        use_poseidon_fluid_normalization=params.get(
            "use_poseidon_fluid_normalization", True
        ),
        magnetic_input_init=params.get("magnetic_input_init", "mean_velocity"),
        magnetic_output_init=params.get("magnetic_output_init", "zero"),
        velocity_residual=params.get("velocity_residual", False),
        velocity_residual_scale=params.get("velocity_residual_scale", 1.0),
        magnetic_residual=params.get("magnetic_residual", True),
        magnetic_residual_scale=params.get("magnetic_residual_scale", 1.0),
        log_re_mean=re_config.get("log_re_mean", 2.9756),
        log_re_std=re_config.get("log_re_std", 0.5417),
        include_rem=re_config.get("include_rem", True),
        re_reference=re_config.get("reference_re", 1000.0),
        re_channel_init=re_config.get("channel_init", "zero"),
    )
