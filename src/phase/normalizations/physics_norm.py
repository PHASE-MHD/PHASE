"""Physics-aware normalization that applies channel-wise scaling."""

import torch
import torch.nn as nn

from .normalization_factory import register_normalization


class PhysicsNormalization(nn.Module):
    """
    Physics-aware normalization using pre-defined scaling factors.

    This normalization applies channel-wise scaling based on domain knowledge
    of the typical magnitudes of physical quantities. Vector-pair channels can
    optionally share one scale so operations like Helmholtz projection happen in
    a consistent vector space after normalization.

    Args:
        input_norm: List of scaling factors for each input channel
        output_norm: List of scaling factors for each output channel
        paired_input_groups: Optional groups of input channels that share a scale
        paired_output_groups: Optional groups of output channels that share a scale
        per_re_input_norms: Optional mapping from Re value to input norm list
        per_re_output_norms: Optional mapping from Re value to output norm list
    """

    def __init__(
        self,
        input_norm=None,
        output_norm=None,
        paired_input_groups=None,
        paired_output_groups=None,
        paired_mode="rms",
        per_re_input_norms=None,
        per_re_output_norms=None,
    ):
        super().__init__()

        self.paired_mode = paired_mode
        self._per_re_input_norms = self._prepare_per_re_norms(
            per_re_input_norms, paired_input_groups
        )
        self._per_re_output_norms = self._prepare_per_re_norms(
            per_re_output_norms, paired_output_groups
        )

        if input_norm is not None:
            input_norm = self._apply_paired_groups(input_norm, paired_input_groups)
            self.register_buffer("input_norm", torch.tensor(input_norm))
        else:
            self.input_norm = None

        if output_norm is not None:
            output_norm = self._apply_paired_groups(output_norm, paired_output_groups)
            self.register_buffer("output_norm", torch.tensor(output_norm))
        else:
            self.output_norm = None

    def _group_scale(self, values):
        values = torch.as_tensor(values, dtype=torch.float32)
        if self.paired_mode == "mean":
            return values.mean()
        if self.paired_mode == "max":
            return values.max()
        if self.paired_mode == "min":
            return values.min()
        if self.paired_mode != "rms":
            raise ValueError("paired_mode must be one of: rms, mean, max, min")
        return torch.sqrt(torch.mean(values.square()))

    def _apply_paired_groups(self, norm_values, paired_groups):
        if not paired_groups:
            return norm_values
        norm = torch.as_tensor(norm_values, dtype=torch.float32).clone()
        for group in paired_groups:
            indices = [int(idx) for idx in group]
            scale = self._group_scale(norm[indices])
            norm[indices] = scale
        return norm.tolist()

    def _prepare_per_re_norms(self, per_re_norms, paired_groups):
        if not per_re_norms:
            return {}
        prepared = {}
        for key, values in per_re_norms.items():
            norm = self._apply_paired_groups(values, paired_groups)
            prepared[str(float(key))] = torch.as_tensor(norm, dtype=torch.float32)
        return prepared

    def _metadata_re_values(self, metadata, batch_size, device):
        if metadata is None or "re" not in metadata:
            return None
        re_values = metadata["re"]
        if not torch.is_tensor(re_values):
            re_values = torch.as_tensor(re_values, dtype=torch.float32, device=device)
        else:
            re_values = re_values.to(device=device, dtype=torch.float32)
        re_values = re_values.reshape(-1)
        if re_values.numel() == 1 and batch_size != 1:
            re_values = re_values.repeat(batch_size)
        return re_values

    def _select_per_re_norm(self, per_re_norms, metadata, x, fallback_norm):
        if not per_re_norms:
            return fallback_norm
        re_values = self._metadata_re_values(metadata, x.shape[0], x.device)
        if re_values is None:
            return fallback_norm

        norms = []
        for re_value in re_values.detach().cpu().tolist():
            key = str(float(re_value))
            if key not in per_re_norms:
                raise KeyError(f"No per-Re normalization entry for Re={re_value}")
            norms.append(per_re_norms[key].to(x.device))
        return torch.stack(norms, dim=0)

    def _apply_norm(self, x, norm, inverse=False):
        if norm is None:
            return x
        if norm.ndim == 1:
            view_shape = [1, -1] + [1] * (x.ndim - 2)
        else:
            view_shape = [norm.shape[0], norm.shape[1]] + [1] * (x.ndim - 2)
        norm = norm.to(x.device).view(*view_shape)
        return x * norm if inverse else x / norm

    def normalize(self, x, metadata=None):
        """Apply normalization to input"""
        input_norm = self.input_norm if self.input_norm is not None else None
        input_norm = self._select_per_re_norm(
            self._per_re_input_norms, metadata, x, input_norm
        )
        return self._apply_norm(x, input_norm, inverse=False)

    def denormalize(self, x, metadata=None):
        """Apply denormalization to output"""
        output_norm = self.output_norm if self.output_norm is not None else None
        output_norm = self._select_per_re_norm(
            self._per_re_output_norms, metadata, x, output_norm
        )
        return self._apply_norm(x, output_norm, inverse=True)

    def to(self, device):
        """Override to method to properly handle device transfers"""
        if self.input_norm is not None:
            self.input_norm = self.input_norm.to(device)
        if self.output_norm is not None:
            self.output_norm = self.output_norm.to(device)
        return self


@register_normalization("physics")
def create_physics_normalization(norm_params):
    """Create a physics-aware normalization using pre-defined scaling factors."""
    return PhysicsNormalization(
        input_norm=norm_params.get("input_norm"),
        output_norm=norm_params.get("output_norm"),
        paired_input_groups=norm_params.get("paired_input_groups"),
        paired_output_groups=norm_params.get("paired_output_groups"),
        paired_mode=norm_params.get("paired_mode", "rms"),
        per_re_input_norms=norm_params.get("per_re_input_norms"),
        per_re_output_norms=norm_params.get("per_re_output_norms"),
    )
