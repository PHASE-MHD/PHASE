"""Poseidon scOT expanded with trainable magnetic channels.

This model is a direct fine-tuning baseline: start from pretrained Poseidon
fluid weights, expand the input/output channel count from [rho, u, v, p] to
[rho, u, v, p, magnetic...], and train the expanded scOT on MHD data. The
public forward API stays compatible with DINOs and returns either [u, v, A] or
[u, v, Bx, By].
"""

from copy import deepcopy
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

POSEIDON_FLUID_MEAN = (0.80, 0.0, 0.0, 0.0)
POSEIDON_FLUID_STD = (0.31, 0.391, 0.356, 0.185)


class PoseidonMHDFinetune(nn.Module):
    """Expanded Poseidon/scOT model for MHD fine-tuning."""

    def __init__(
        self,
        poseidon_model: Optional[str] = "camlab-ethz/Poseidon-T",
        load_pretrained_poseidon: bool = True,
        image_size: int = 128,
        patch_size: int = 4,
        out_channels: int = 3,
        poseidon_input_channel_map: Tuple[int, int] = (1, 2),
        poseidon_output_channel_map: Tuple[int, int] = (1, 2),
        magnetic_channel_index: int = 4,
        magnetic_channel_indices: Optional[Sequence[int]] = None,
        use_poseidon_fluid_normalization: bool = True,
        magnetic_input_init: str = "zero",
        magnetic_output_init: str = "zero",
        velocity_residual: bool = False,
        velocity_residual_scale: float = 1.0,
        magnetic_residual: bool = False,
        magnetic_residual_scale: float = 1.0,
        helmholtz_projection: bool = False,
        project_velocity: bool = True,
        project_magnetic: bool = True,
        helmholtz_domain_size_x: float = 1.0,
        helmholtz_domain_size_y: float = 1.0,
        freeze_pretrained_backbone: bool = False,
        train_patch_embedding: bool = True,
        train_patch_recovery: bool = True,
        fallback_poseidon_num_channels: int = 5,
        fallback_poseidon_num_out_channels: int = 5,
        fallback_poseidon_embed_dim: int = 48,
        fallback_poseidon_depths: Sequence[int] = (2, 2, 2, 2),
        fallback_poseidon_num_heads: Sequence[int] = (3, 6, 12, 24),
        fallback_poseidon_skip_connections: Optional[Sequence[int]] = None,
        window_size: int = 7,
        skip_connections: Sequence[int] = (2, 2, 2, 0),
        **kwargs,
    ):
        super().__init__()
        if out_channels not in (3, 4):
            raise ValueError(
                "PoseidonMHDFinetune expects out_channels=3 ([u, v, A]) "
                "or out_channels=4 ([u, v, Bx, By])."
            )

        try:
            from scOT.model import ScOT, ScOTConfig
        except ImportError as exc:
            raise ImportError(
                "Could not import Poseidon scOT modules. Install the local poseidon "
                "package or make sure it is on PYTHONPATH."
            ) from exc

        self.poseidon_input_channel_map = tuple(poseidon_input_channel_map)
        self.poseidon_output_channel_map = tuple(poseidon_output_channel_map)
        self.out_channels = int(out_channels)
        self.num_magnetic_channels = self.out_channels - 2
        if magnetic_channel_indices is None:
            start = int(magnetic_channel_index)
            magnetic_channel_indices = tuple(
                range(start, start + self.num_magnetic_channels)
            )
        self.magnetic_channel_indices = tuple(int(idx) for idx in magnetic_channel_indices)
        if len(self.magnetic_channel_indices) != self.num_magnetic_channels:
            raise ValueError(
                f"Expected {self.num_magnetic_channels} magnetic channel indices "
                f"for out_channels={self.out_channels}, got "
                f"{self.magnetic_channel_indices}."
            )
        self.magnetic_channel_index = self.magnetic_channel_indices[0]
        self.use_poseidon_fluid_normalization = use_poseidon_fluid_normalization
        self.velocity_residual = velocity_residual
        self.velocity_residual_scale = velocity_residual_scale
        self.magnetic_residual = magnetic_residual
        self.magnetic_residual_scale = magnetic_residual_scale
        self.helmholtz_projection = bool(helmholtz_projection)
        self.project_velocity = bool(project_velocity)
        self.project_magnetic = bool(project_magnetic)
        self.helmholtz_domain_size_x = float(helmholtz_domain_size_x)
        self.helmholtz_domain_size_y = float(helmholtz_domain_size_y)
        self._optimizer_hooks_registered = False

        if load_pretrained_poseidon:
            if poseidon_model is None:
                raise ValueError(
                    "poseidon_model must be set when load_pretrained_poseidon=True."
                )
            pretrained = ScOT.from_pretrained(poseidon_model)
            expanded_config = deepcopy(pretrained.config)
            expanded_config.num_channels = max(
                int(pretrained.config.num_channels) + self.num_magnetic_channels,
                max(self.magnetic_channel_indices) + 1,
            )
            expanded_config.num_out_channels = max(
                int(pretrained.config.num_out_channels)
                + self.num_magnetic_channels,
                max(self.magnetic_channel_indices) + 1,
            )
            self.poseidon = ScOT(expanded_config)
            self._copy_expanded_poseidon_weights(
                pretrained,
                magnetic_input_init=magnetic_input_init,
                magnetic_output_init=magnetic_output_init,
            )
        else:
            if fallback_poseidon_skip_connections is None:
                fallback_poseidon_skip_connections = skip_connections
            config = ScOTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=fallback_poseidon_num_channels,
                num_out_channels=fallback_poseidon_num_out_channels,
                embed_dim=fallback_poseidon_embed_dim,
                depths=list(fallback_poseidon_depths),
                num_heads=list(fallback_poseidon_num_heads),
                skip_connections=list(fallback_poseidon_skip_connections),
                window_size=window_size,
                hidden_act="gelu",
                use_conditioning=True,
                residual_model="convnext",
            )
            self.poseidon = ScOT(config)

        self.poseidon_num_channels = int(self.poseidon.config.num_channels)
        self.poseidon_num_out_channels = int(self.poseidon.config.num_out_channels)
        self.pretrained_num_channels = (
            int(pretrained.config.num_channels) if load_pretrained_poseidon else 4
        )
        self.pretrained_num_out_channels = (
            int(pretrained.config.num_out_channels) if load_pretrained_poseidon else 4
        )
        if freeze_pretrained_backbone:
            for param in self.poseidon.parameters():
                param.requires_grad_(False)
            if train_patch_embedding:
                for param in self.poseidon.embeddings.patch_embeddings.parameters():
                    param.requires_grad_(True)
            if train_patch_recovery:
                for param in self.poseidon.patch_recovery.parameters():
                    param.requires_grad_(True)

    def _copy_expanded_poseidon_weights(
        self,
        pretrained: nn.Module,
        magnetic_input_init: str = "zero",
        magnetic_output_init: str = "zero",
    ) -> None:
        pretrained_state = pretrained.state_dict()
        expanded_state = self.poseidon.state_dict()

        for key, value in pretrained_state.items():
            if key not in expanded_state:
                continue
            target = expanded_state[key]
            if target.shape == value.shape:
                target.copy_(value)
            elif key == "embeddings.patch_embeddings.projection.weight":
                target[:, : value.shape[1]].copy_(value)
                if target.shape[1] > value.shape[1]:
                    if magnetic_input_init == "mean_velocity":
                        velocity_slice = value[:, 1:3].mean(dim=1, keepdim=True)
                        target[:, value.shape[1] :].copy_(velocity_slice)
                    elif magnetic_input_init == "mean_all":
                        target[:, value.shape[1] :].copy_(
                            value.mean(dim=1, keepdim=True)
                        )
                    elif magnetic_input_init == "zero":
                        target[:, value.shape[1] :].zero_()
                    else:
                        raise ValueError(
                            "magnetic_input_init must be 'zero', 'mean_velocity', "
                            "or 'mean_all'."
                        )
            elif key == "patch_recovery.projection.weight":
                target[:, : value.shape[1]].copy_(value)
                if target.shape[1] > value.shape[1]:
                    if magnetic_output_init == "mean_velocity":
                        velocity_slice = value[:, 1:3].mean(dim=1, keepdim=True)
                        target[:, value.shape[1] :].copy_(velocity_slice)
                    elif magnetic_output_init == "mean_all":
                        target[:, value.shape[1] :].copy_(
                            value.mean(dim=1, keepdim=True)
                        )
                    elif magnetic_output_init == "zero":
                        target[:, value.shape[1] :].zero_()
                    else:
                        raise ValueError(
                            "magnetic_output_init must be 'zero', 'mean_velocity', "
                            "or 'mean_all'."
                        )
            elif key == "patch_recovery.projection.bias":
                target[: value.shape[0]].copy_(value)
                target[value.shape[0] :].zero_()
            elif key == "patch_recovery.mixup.weight":
                target.zero_()
                target[: value.shape[0], : value.shape[1]].copy_(value)
                if target.shape[0] > value.shape[0] and target.shape[1] > value.shape[1]:
                    extra = min(target.shape[0] - value.shape[0], target.shape[1] - value.shape[1])
                    center_x = target.shape[-2] // 2
                    center_y = target.shape[-1] // 2
                    for i in range(extra):
                        target[value.shape[0] + i, value.shape[1] + i, center_x, center_y] = 1.0

        self.poseidon.load_state_dict(expanded_state)

    def _register_old_slice_gradient_scaling(
        self,
        old_to_new_lr_ratio: float,
        magnetic_output_lr_ratio: float = 1.0,
    ) -> None:
        """Scale boundary tensor slices to give copied and new channels different effective LRs."""
        if self._optimizer_hooks_registered:
            return
        old_ratio = float(old_to_new_lr_ratio)
        output_ratio = float(magnetic_output_lr_ratio)

        def scale_patch_embedding_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[:, : self.pretrained_num_channels] *= old_ratio
            return grad

        def scale_recovery_projection_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            if grad.shape[0] == self.poseidon_num_out_channels:
                grad[: self.pretrained_num_out_channels] *= old_ratio
                grad[self.pretrained_num_out_channels :] *= output_ratio
            else:
                grad[:, : self.pretrained_num_out_channels] *= old_ratio
                grad[:, self.pretrained_num_out_channels :] *= output_ratio
            return grad

        def scale_recovery_mixup_grad(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[: self.pretrained_num_out_channels] *= old_ratio
            grad[self.pretrained_num_out_channels :] *= output_ratio
            return grad

        self.poseidon.embeddings.patch_embeddings.projection.weight.register_hook(
            scale_patch_embedding_grad
        )
        self.poseidon.patch_recovery.projection.weight.register_hook(
            scale_recovery_projection_grad
        )
        if self.poseidon.patch_recovery.projection.bias is not None:
            self.poseidon.patch_recovery.projection.bias.register_hook(
                scale_recovery_projection_grad
            )
        self.poseidon.patch_recovery.mixup.weight.register_hook(
            scale_recovery_mixup_grad
        )
        self._optimizer_hooks_registered = True

    def get_optimizer_parameters(self, optimizer_params):
        """Return parameter groups with a larger LR on expanded magnetic channels.

        The new magnetic input/output channel slices live inside convolution tensors
        that also contain copied Poseidon weights. Those tensors are assigned the
        new-channel LR, while hooks scale copied-channel gradients by
        pretrained_lr / new_lr so their effective update remains small.
        """
        param_group_config = optimizer_params.get("param_groups", {})
        if not param_group_config or not param_group_config.get("enabled", False):
            return self.parameters()

        pretrained_lr = param_group_config.get(
            "pretrained_lr", optimizer_params.get("lr", 1e-5)
        )
        new_lr = param_group_config.get("new_lr", optimizer_params.get("lr", 1e-5))
        magnetic_output_lr = param_group_config.get("magnetic_output_lr", new_lr)
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
        pretrained_params = []
        boundary_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith(boundary_names):
                boundary_params.append(param)
            else:
                pretrained_params.append(param)

        if new_lr != 0:
            self._register_old_slice_gradient_scaling(
                pretrained_lr / new_lr,
                magnetic_output_lr / new_lr,
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
        return groups

    def adapt_mhd_to_poseidon(self, fields: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = fields.shape
        adapted = fields.new_zeros(
            batch_size, self.poseidon_num_channels, height, width
        )

        if self.use_poseidon_fluid_normalization and self.poseidon_num_channels >= 4:
            mean = fields.new_tensor(POSEIDON_FLUID_MEAN)
            std = fields.new_tensor(POSEIDON_FLUID_STD)
            raw_fill = fields.new_zeros(batch_size, 4, height, width)
            raw_fill[:, 0] = 1.0
            raw_fill[:, 3] = 0.0
            adapted[:, :4] = (raw_fill - mean.view(1, 4, 1, 1)) / std.view(1, 4, 1, 1)
            adapted[:, self.poseidon_input_channel_map[0]] = fields[:, 0] / std[
                self.poseidon_input_channel_map[0]
            ]
            adapted[:, self.poseidon_input_channel_map[1]] = fields[:, 1] / std[
                self.poseidon_input_channel_map[1]
            ]
        else:
            adapted[:, self.poseidon_input_channel_map[0]] = fields[:, 0]
            adapted[:, self.poseidon_input_channel_map[1]] = fields[:, 1]

        adapted[:, list(self.magnetic_channel_indices)] = fields[
            :, 2 : 2 + self.num_magnetic_channels
        ]
        return adapted

    def _project_pair_helmholtz(
        self, fields: torch.Tensor, x_channel: int, y_channel: int
    ) -> torch.Tensor:
        """Project one vector channel pair to divergence-free with spectral Helmholtz."""
        orig_dtype = fields.dtype
        qx = fields[:, x_channel].double()
        qy = fields[:, y_channel].double()
        height, width = qx.shape[-2:]
        device = qx.device

        kx = 2.0 * torch.pi * torch.fft.fftfreq(
            height,
            d=self.helmholtz_domain_size_x / height,
            device=device,
            dtype=qx.dtype,
        ).reshape(1, height, 1)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(
            width,
            d=self.helmholtz_domain_size_y / width,
            device=device,
            dtype=qx.dtype,
        ).reshape(1, 1, width)

        qx_hat = torch.fft.fft2(qx, dim=(-2, -1))
        qy_hat = torch.fft.fft2(qy, dim=(-2, -1))
        k2 = kx.square() + ky.square()
        k2[..., 0, 0] = 1.0

        k_dot_q = kx * qx_hat + ky * qy_hat
        qx_hat_proj = qx_hat - kx * k_dot_q / k2
        qy_hat_proj = qy_hat - ky * k_dot_q / k2

        if height % 2 == 0:
            qx_hat_proj[:, height // 2, :] = 0.0
            qy_hat_proj[:, height // 2, :] = 0.0
        if width % 2 == 0:
            qx_hat_proj[:, :, width // 2] = 0.0
            qy_hat_proj[:, :, width // 2] = 0.0

        qx_proj = torch.fft.ifft2(qx_hat_proj, dim=(-2, -1)).real
        qy_proj = torch.fft.ifft2(qy_hat_proj, dim=(-2, -1)).real

        projected = fields.clone()
        projected[:, x_channel] = qx_proj.to(orig_dtype)
        projected[:, y_channel] = qy_proj.to(orig_dtype)
        return projected

    def project_fields_helmholtz(self, fields: torch.Tensor) -> torch.Tensor:
        """Project configured vector pairs before losses or rollout consumers see them."""
        if not self.helmholtz_projection:
            return fields
        if self.project_velocity:
            if fields.shape[1] < 2:
                raise ValueError(
                    "Velocity Helmholtz projection expects channels [u, v]."
                )
            fields = self._project_pair_helmholtz(fields, 0, 1)
        if self.project_magnetic and self.out_channels == 4:
            if fields.shape[1] < 4:
                raise ValueError(
                    "Magnetic Helmholtz projection expects channels [Bx, By]."
                )
            fields = self._project_pair_helmholtz(fields, 2, 3)
        return fields

    def extract_mhd_prediction(
        self, prediction: torch.Tensor, initial_fields: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        velocity = prediction[:, list(self.poseidon_output_channel_map)]
        if self.use_poseidon_fluid_normalization:
            std = prediction.new_tensor(
                [
                    POSEIDON_FLUID_STD[self.poseidon_output_channel_map[0]],
                    POSEIDON_FLUID_STD[self.poseidon_output_channel_map[1]],
                ]
            )
            mean = prediction.new_tensor(
                [
                    POSEIDON_FLUID_MEAN[self.poseidon_output_channel_map[0]],
                    POSEIDON_FLUID_MEAN[self.poseidon_output_channel_map[1]],
                ]
            )
            velocity = velocity * std.view(1, 2, 1, 1) + mean.view(1, 2, 1, 1)
        if self.velocity_residual:
            if initial_fields is None:
                raise ValueError(
                    "initial_fields must be provided when velocity_residual=True."
                )
            velocity = initial_fields[:, :2] + self.velocity_residual_scale * velocity

        magnetic = prediction[:, list(self.magnetic_channel_indices)]
        if self.magnetic_residual:
            if initial_fields is None:
                raise ValueError(
                    "initial_fields must be provided when magnetic_residual=True."
                )
            magnetic = (
                initial_fields[:, 2 : 2 + self.num_magnetic_channels]
                + self.magnetic_residual_scale * magnetic
            )
        fields = torch.cat([velocity, magnetic], dim=1)
        return self.project_fields_helmholtz(fields)

    def forward_transition(
        self, x: torch.Tensor, time: torch.Tensor
    ) -> torch.Tensor:
        pixel_values = self.adapt_mhd_to_poseidon(x)
        output = self.poseidon(pixel_values=pixel_values, time=time)
        return self.extract_mhd_prediction(output.output, initial_fields=x)

    def forward(self, x: torch.Tensor, time: Optional[torch.Tensor] = None):
        if x.dim() == 4:
            if time is None:
                time = x.new_ones(x.shape[0])
            return self.forward_transition(x, time)

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
            predictions.append(self.forward_transition(transition_input, step_time))
        return torch.stack(predictions, dim=2)


def create_poseidon_mhd_finetune(params):
    return PoseidonMHDFinetune(
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
        magnetic_input_init=params.get("magnetic_input_init", "zero"),
        magnetic_output_init=params.get("magnetic_output_init", "zero"),
        velocity_residual=params.get("velocity_residual", False),
        velocity_residual_scale=params.get("velocity_residual_scale", 1.0),
        magnetic_residual=params.get("magnetic_residual", False),
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
    )
