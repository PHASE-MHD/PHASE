"""Re-conditioned Poseidon MHD fine-tuning model.

The base model is the expanded Poseidon MHD fine-tune path. Reynolds number can
condition the model through either final-field FiLM modulation or zero-initialized
deep residual adapters attached to the scOT encoder/decoder blocks. The deep
adapter path keeps a warm-started single-Re operator intact at initialization,
then lets Re/ReM-dependent corrections act throughout the transformer.
"""

from typing import Optional

import torch
import torch.nn as nn

from .scot_mhd import PoseidonMHDFinetune


class ReFiLMConditioner(nn.Module):
    """Small MLP that maps log-Re features to channel-wise scale and shift."""

    def __init__(
        self,
        out_channels: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 2,
        log_re_mean: float = 2.9756,
        log_re_std: float = 0.5417,
        include_rem: bool = True,
        residual_scale: float = 1.0,
        zero_init: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.include_rem = include_rem
        self.residual_scale = residual_scale
        self.register_buffer("log_re_mean", torch.tensor(float(log_re_mean)))
        self.register_buffer("log_re_std", torch.tensor(float(log_re_std)))

        in_dim = 2 if include_rem else 1
        layers = []
        current_dim = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 2 * out_channels))
        self.net = nn.Sequential(*layers)

        if zero_init:
            final = self.net[-1]
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)

    def _normalize_re(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(1e-12)
        return (torch.log10(value) - self.log_re_mean) / self.log_re_std.clamp_min(1e-6)

    def forward(
        self, re: torch.Tensor, rem: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if rem is None:
            rem = re
        features = [self._normalize_re(re).view(-1, 1)]
        if self.include_rem:
            features.append(self._normalize_re(rem).view(-1, 1))
        conditioning = torch.cat(features, dim=-1)
        gamma, beta = self.net(conditioning).chunk(2, dim=-1)
        return gamma, beta


class ReResidualAdapter(nn.Module):
    """Zero-initialized residual adapter conditioned on log-Re/log-ReM."""

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
        self.include_rem = include_rem
        self.adapter_scale = float(adapter_scale)
        self.gate_type = gate_type
        self.register_buffer("log_re_mean", torch.tensor(float(log_re_mean)))
        self.register_buffer("log_re_std", torch.tensor(float(log_re_std)))

        bottleneck_dim = int(bottleneck_dim)
        self.norm = nn.LayerNorm(self.dim)
        self.down = nn.Linear(self.dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, self.dim)

        in_dim = 2 if include_rem else 1
        out_dim = self.dim if gate_type == "channel" else 1
        layers = []
        current_dim = in_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.extend([nn.Linear(current_dim, hidden_dim), nn.GELU()])
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.gate_net = nn.Sequential(*layers)

        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        final = self.gate_net[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def _normalize_re(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float().clamp_min(1e-12)
        return (torch.log10(value) - self.log_re_mean) / self.log_re_std.clamp_min(1e-6)

    def _conditioning_features(
        self, re: torch.Tensor, rem: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if rem is None:
            rem = re
        features = [self._normalize_re(re).view(-1, 1)]
        if self.include_rem:
            features.append(self._normalize_re(rem).view(-1, 1))
        return torch.cat(features, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        re: torch.Tensor,
        rem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conditioning = self._conditioning_features(re, rem)
        gate = 1.0 + self.gate_net(conditioning).to(hidden_states.dtype)
        gate = gate.view(hidden_states.shape[0], 1, -1)
        update = self.up(self.act(self.down(self.norm(hidden_states))))
        return hidden_states + self.adapter_scale * gate * update


class PoseidonMHDReFinetune(PoseidonMHDFinetune):
    """Poseidon MHD fine-tune model with optional Re/ReM conditioning."""

    accepts_conditioning = True

    def __init__(
        self,
        re_conditioning_enabled: bool = True,
        re_conditioning_type: str = "output_film",
        re_hidden_dim: int = 128,
        re_num_layers: int = 2,
        log_re_mean: float = 2.9756,
        log_re_std: float = 0.5417,
        include_rem: bool = True,
        re_residual_scale: float = 1.0,
        re_reference: float = 1000.0,
        deep_adapter_config: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        valid_types = {"output_film", "deep_adapter", "deep_adapter_output_film"}
        if re_conditioning_type not in valid_types:
            raise ValueError(
                "re_conditioning_type must be one of "
                f"{sorted(valid_types)}, got {re_conditioning_type!r}."
            )
        self.re_conditioning_enabled = re_conditioning_enabled
        self.re_conditioning_type = re_conditioning_type
        self.re_reference = float(re_reference)
        self.output_re_conditioning_enabled = (
            re_conditioning_enabled
            and re_conditioning_type in {"output_film", "deep_adapter_output_film"}
        )
        self.deep_re_conditioning_enabled = (
            re_conditioning_enabled
            and re_conditioning_type in {"deep_adapter", "deep_adapter_output_film"}
        )
        if self.output_re_conditioning_enabled:
            self.re_conditioner = ReFiLMConditioner(
                out_channels=self.out_channels,
                hidden_dim=re_hidden_dim,
                num_layers=re_num_layers,
                log_re_mean=log_re_mean,
                log_re_std=log_re_std,
                include_rem=include_rem,
                residual_scale=re_residual_scale,
                zero_init=True,
            )
        else:
            self.re_conditioner = None

        self._active_re = None
        self._active_rem = None
        self._conditioning_context_active = False
        self.deep_re_adapters = nn.ModuleList()
        self._deep_re_adapter_handles = []
        if self.deep_re_conditioning_enabled:
            self._install_deep_re_adapters(
                deep_adapter_config or {},
                log_re_mean=log_re_mean,
                log_re_std=log_re_std,
                include_rem=include_rem,
            )

    def expected_warm_start_missing_keys(self) -> set[str]:
        """State introduced after the single-Re warm-start checkpoint."""
        prefixes = ("re_conditioner.", "deep_re_adapters.")
        return {key for key in self.state_dict() if key.startswith(prefixes)}

    def _iter_scot_blocks(self, target: str):
        if target in {"all", "encoder"}:
            for stage_idx, stage in enumerate(self.poseidon.encoder.layers):
                for block_idx, block in enumerate(stage.blocks):
                    yield f"encoder.{stage_idx}.{block_idx}", stage.dim, block
        if target in {"all", "decoder"}:
            for stage_idx, stage in enumerate(self.poseidon.decoder.layers):
                for block_idx, block in enumerate(stage.blocks):
                    yield f"decoder.{stage_idx}.{block_idx}", stage.dim, block

    def _prepare_conditioning_tensor(
        self,
        value: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        default_value: float,
    ) -> torch.Tensor:
        if value is None:
            value = hidden_states.new_full((hidden_states.shape[0],), default_value)
        else:
            value = value.to(hidden_states.device, dtype=hidden_states.dtype).view(-1)
            if value.numel() == 1 and hidden_states.shape[0] != 1:
                value = value.expand(hidden_states.shape[0])
        if value.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "Re conditioning batch size mismatch: "
                f"got {value.shape[0]} values for hidden batch {hidden_states.shape[0]}."
            )
        return value

    def _make_deep_adapter_hook(self, adapter: ReResidualAdapter):
        def hook(_module, _inputs, output):
            if (
                not self.deep_re_conditioning_enabled
                or not self._conditioning_context_active
            ):
                return output
            hidden_states = output[0] if isinstance(output, tuple) else output
            re = self._prepare_conditioning_tensor(
                self._active_re, hidden_states, self.re_reference
            )
            rem = self._prepare_conditioning_tensor(
                self._active_rem, hidden_states, self.re_reference
            )
            adapted = adapter(hidden_states, re=re, rem=rem)
            if isinstance(output, tuple):
                return (adapted,) + output[1:]
            return adapted

        return hook

    def _install_deep_re_adapters(
        self,
        config: dict,
        log_re_mean: float,
        log_re_std: float,
        include_rem: bool,
    ) -> None:
        target = config.get("target", "all")
        bottleneck_dim = config.get("bottleneck_dim", 64)
        hidden_dim = config.get("hidden_dim", 128)
        num_layers = config.get("num_layers", 2)
        adapter_scale = config.get("adapter_scale", 1.0)
        gate_type = config.get("gate_type", "channel")

        installed = 0
        for _name, dim, block in self._iter_scot_blocks(target):
            adapter = ReResidualAdapter(
                dim=dim,
                bottleneck_dim=bottleneck_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                log_re_mean=log_re_mean,
                log_re_std=log_re_std,
                include_rem=include_rem,
                adapter_scale=adapter_scale,
                gate_type=gate_type,
            )
            self.deep_re_adapters.append(adapter)
            self._deep_re_adapter_handles.append(
                block.register_forward_hook(self._make_deep_adapter_hook(adapter))
            )
            installed += 1
        if installed == 0:
            raise ValueError(f"No scOT blocks matched deep adapter target={target!r}.")

    def apply_re_conditioning(
        self,
        prediction: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        rem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.output_re_conditioning_enabled:
            return prediction
        if re is None:
            re = prediction.new_full((prediction.shape[0],), self.re_reference)
        else:
            re = re.to(prediction.device, dtype=prediction.dtype).view(-1)
            if re.numel() == 1 and prediction.shape[0] != 1:
                re = re.expand(prediction.shape[0])
        if rem is not None:
            rem = rem.to(prediction.device, dtype=prediction.dtype).view(-1)
            if rem.numel() == 1 and prediction.shape[0] != 1:
                rem = rem.expand(prediction.shape[0])

        gamma, beta = self.re_conditioner(re, rem)
        gamma = gamma.to(prediction.dtype).view(prediction.shape[0], -1, 1, 1)
        beta = beta.to(prediction.dtype).view(prediction.shape[0], -1, 1, 1)
        return prediction + self.re_conditioner.residual_scale * (
            prediction * gamma + beta
        )

    def get_optimizer_parameters(self, optimizer_params):
        param_group_config = optimizer_params.get("param_groups", {})
        if not param_group_config or not param_group_config.get("enabled", False):
            return self.parameters()

        pretrained_lr = param_group_config.get(
            "pretrained_lr", optimizer_params.get("lr", 1e-5)
        )
        new_lr = param_group_config.get("new_lr", optimizer_params.get("lr", 1e-5))
        weight_decay = optimizer_params.get("weight_decay", 0.0)
        new_weight_decay = param_group_config.get("new_weight_decay", weight_decay)
        pretrained_weight_decay = param_group_config.get(
            "pretrained_weight_decay", weight_decay
        )

        boundary_names = (
            "poseidon.embeddings.patch_embeddings.projection",
            "poseidon.patch_recovery.projection",
            "poseidon.patch_recovery.mixup",
        )
        adapter_names = ("re_conditioner", "deep_re_adapters")
        boundary_group = param_group_config.get("boundary_group", "new")
        freeze_pretrained = param_group_config.get("freeze_pretrained", False)
        train_boundary_when_frozen = param_group_config.get(
            "train_boundary_when_frozen", False
        )
        pretrained_params = []
        boundary_params = []
        adapter_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_adapter = name.startswith(adapter_names)
            is_boundary = name.startswith(boundary_names)
            if freeze_pretrained and not is_adapter and not (
                train_boundary_when_frozen and is_boundary
            ):
                param.requires_grad_(False)
                continue
            if is_adapter:
                adapter_params.append(param)
            elif is_boundary and boundary_group == "new":
                boundary_params.append(param)
            else:
                pretrained_params.append(param)

        if boundary_params and new_lr != 0:
            self._register_old_slice_gradient_scaling(
                pretrained_lr / new_lr,
                param_group_config.get("magnetic_output_lr", new_lr) / new_lr,
            )

        groups = []
        if pretrained_params:
            groups.append(
                {
                    "params": pretrained_params,
                    "lr": pretrained_lr,
                    "weight_decay": pretrained_weight_decay,
                    "name": "pretrained",
                }
            )
        if boundary_params:
            groups.append(
                {
                    "params": boundary_params,
                    "lr": new_lr,
                    "weight_decay": new_weight_decay,
                    "name": "expanded_magnetic_boundary",
                }
            )
        if adapter_params:
            groups.append(
                {
                    "params": adapter_params,
                    "lr": new_lr,
                    "weight_decay": new_weight_decay,
                    "name": "re_conditioning_adapters",
                }
            )
        return groups

    def forward_transition(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        re: Optional[torch.Tensor] = None,
        rem: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        previous_re = self._active_re
        previous_rem = self._active_rem
        previous_context_active = self._conditioning_context_active
        self._active_re = re
        self._active_rem = rem if rem is not None else re
        self._conditioning_context_active = True
        try:
            prediction = super().forward_transition(x, time)
            prediction = self.apply_re_conditioning(prediction, re=re, rem=rem)
            # Conditioning can reintroduce divergence after the base projection.
            return self.project_fields_helmholtz(prediction)
        finally:
            self._active_re = previous_re
            self._active_rem = previous_rem
            self._conditioning_context_active = previous_context_active

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
                "Expected x with shape [B, out_channels, H, W] or [B, C, T, H, W], "
                f"got {tuple(x.shape)}."
            )
        expected_input_channels = 3 + self.out_channels
        if x.shape[1] < expected_input_channels:
            raise ValueError(
                f"5D compatibility mode expects at least {expected_input_channels} "
                "channels: [t, x, y] plus repeated initial condition fields."
            )

        predictions = []
        for time_index in range(x.shape[2]):
            transition_input = x[:, 3 : 3 + self.out_channels, time_index]
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


def create_poseidon_mhd_re_finetune(params):
    """Build the locked gated-adapter Re/Rm-conditioning model."""
    re_config = params.get("re_conditioning", {})
    return PoseidonMHDReFinetune(
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
        magnetic_channel_indices=params.get("magnetic_channel_indices", None),
        use_poseidon_fluid_normalization=params.get(
            "use_poseidon_fluid_normalization", True
        ),
        magnetic_input_init=params.get("magnetic_input_init", "mean_velocity"),
        magnetic_output_init=params.get("magnetic_output_init", "zero"),
        velocity_residual=params.get("velocity_residual", False),
        velocity_residual_scale=params.get("velocity_residual_scale", 1.0),
        magnetic_residual=params.get("magnetic_residual", True),
        magnetic_residual_scale=params.get("magnetic_residual_scale", 1.0),
        helmholtz_projection=params.get("helmholtz_projection", False),
        project_velocity=params.get("project_velocity", True),
        project_magnetic=params.get("project_magnetic", True),
        helmholtz_domain_size_x=params.get("helmholtz_domain_size_x", 1.0),
        helmholtz_domain_size_y=params.get("helmholtz_domain_size_y", 1.0),
        freeze_pretrained_backbone=params.get("freeze_pretrained_backbone", False),
        train_patch_embedding=params.get("train_patch_embedding", True),
        train_patch_recovery=params.get("train_patch_recovery", True),
        fallback_poseidon_num_channels=params.get("fallback_poseidon_num_channels", 5),
        fallback_poseidon_num_out_channels=params.get(
            "fallback_poseidon_num_out_channels", 5
        ),
        fallback_poseidon_embed_dim=params.get("fallback_poseidon_embed_dim", 48),
        fallback_poseidon_depths=params.get("fallback_poseidon_depths", [2, 2, 2, 2]),
        fallback_poseidon_num_heads=params.get(
            "fallback_poseidon_num_heads", [3, 6, 12, 24]
        ),
        fallback_poseidon_skip_connections=params.get(
            "fallback_poseidon_skip_connections", None
        ),
        window_size=params.get("window_size", 7),
        skip_connections=params.get("skip_connections", [2, 2, 2, 0]),
        re_conditioning_enabled=re_config.get("enabled", True),
        re_conditioning_type=re_config.get("type", "output_film"),
        re_hidden_dim=re_config.get("hidden_dim", 128),
        re_num_layers=re_config.get("num_layers", 2),
        log_re_mean=re_config.get("log_re_mean", 2.9756),
        log_re_std=re_config.get("log_re_std", 0.5417),
        include_rem=re_config.get("include_rem", True),
        re_residual_scale=re_config.get("residual_scale", 1.0),
        re_reference=re_config.get("reference_re", 1000.0),
        deep_adapter_config=re_config.get("deep_adapter", {}),
    )
