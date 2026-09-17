"""Min-max normalization that scales data to a specified range."""

import torch
import torch.nn as nn

from .normalization_factory import register_normalization


class MinMaxNormalization(nn.Module):
    """
    Min-max normalization that scales data to a specified range.

    This normalization scales each channel independently to a specified range
    based on the provided per-channel min/max values.

    Args:
        min_val: Pre-computed minimum values per channel
        max_val: Pre-computed maximum values per channel
        feature_range: Tuple of (min, max) for the output range
        eps: Small value to avoid division by zero
    """

    def __init__(self, min_val=None, max_val=None, feature_range=(0, 1), eps=1e-8):
        super().__init__()
        self.feature_range = feature_range
        self.eps = eps

        # Register min and max values as buffers
        if min_val is not None:
            self.register_buffer("min_val", torch.tensor(min_val))
        else:
            self.min_val = None

        if max_val is not None:
            self.register_buffer("max_val", torch.tensor(max_val))
        else:
            self.max_val = None

        # # Debug prints
        # print(f"minval:{self.min_val}")
        # print(f"minval:{self.max_val}")
        # print(f"range:{self.feature_range}")

    def normalize(self, x):
        """Apply per-channel normalization to input"""
        if self.min_val is None or self.max_val is None:
            return x

        min_val = self.min_val.to(x.device).view(1, -1, 1, 1, 1)
        max_val = self.max_val.to(x.device).view(1, -1, 1, 1, 1)

        x_std = (x - min_val) / (max_val - min_val + self.eps)

        return (
            x_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )

    def denormalize(self, x):
        """Apply per-channel denormalization to output"""
        if self.min_val is None or self.max_val is None:
            return x

        min_val = self.min_val.to(x.device).view(1, -1, 1, 1, 1)
        max_val = self.max_val.to(x.device).view(1, -1, 1, 1, 1)

        x_std = (x - self.feature_range[0]) / (
            self.feature_range[1] - self.feature_range[0]
        )

        return x_std * (max_val - min_val + self.eps) + min_val

    def to(self, device):
        """Override to method to properly handle device transfers"""
        if self.min_val is not None:
            self.min_val = self.min_val.to(device)
        if self.max_val is not None:
            self.max_val = self.max_val.to(device)
        return self


class PairedMinMaxNormalization(MinMaxNormalization):
    """
    Min-max normalization with shared scales for vector-component channel groups.

    This keeps vector components such as (ux, uy) and (Bx, By) on the same
    affine scale, so a divergence-free constraint applied in normalized units
    remains divergence-free after denormalization.
    """

    def __init__(
        self,
        min_val=None,
        max_val=None,
        feature_range=(0, 1),
        channel_groups=((0, 1), (2, 3)),
        eps=1e-8,
    ):
        min_val, max_val = self._share_group_ranges(min_val, max_val, channel_groups)
        super().__init__(
            min_val=min_val,
            max_val=max_val,
            feature_range=feature_range,
            eps=eps,
        )
        self.channel_groups = channel_groups

    @staticmethod
    def _share_group_ranges(min_val, max_val, channel_groups):
        if min_val is None or max_val is None:
            return min_val, max_val

        min_tensor = torch.tensor(min_val).clone()
        max_tensor = torch.tensor(max_val).clone()

        for group in channel_groups:
            group = list(group)
            if not group:
                continue
            if max(group) >= min_tensor.numel() or max(group) >= max_tensor.numel():
                continue

            group_min = min_tensor[group].min()
            group_max = max_tensor[group].max()
            min_tensor[group] = group_min
            max_tensor[group] = group_max

        return min_tensor.tolist(), max_tensor.tolist()


class PerRePairedMinMaxNormalization(nn.Module):
    """
    Re-dependent paired min-max normalization.

    Training Re values use explicit per-Re statistics. Re values not present in
    the table use linear interpolation/extrapolation in log(Re), matching the
    continuous FiLM/Re conditioning used by the diffusion model.
    """

    def __init__(
        self,
        stats_by_re=None,
        feature_range=(0, 1),
        channel_groups=((0, 1), (2, 3)),
        default_re=None,
        eps=1e-8,
    ):
        super().__init__()
        if not stats_by_re:
            raise ValueError("PerRePairedMinMaxNormalization requires stats_by_re")

        self.feature_range = feature_range
        self.channel_groups = channel_groups
        self.default_re = default_re
        self.eps = eps

        items = sorted((float(re_value), stats) for re_value, stats in stats_by_re.items())
        re_values = []
        min_values = []
        max_values = []
        for re_value, stats in items:
            min_val, max_val = PairedMinMaxNormalization._share_group_ranges(
                stats.get("min_val"), stats.get("max_val"), channel_groups
            )
            re_values.append(re_value)
            min_values.append(min_val)
            max_values.append(max_val)

        if len(re_values) < 1:
            raise ValueError("At least one Re statistics entry is required")

        self.register_buffer("re_values", torch.tensor(re_values, dtype=torch.float32))
        self.register_buffer("log_re_values", torch.log(self.re_values))
        self.register_buffer("min_values", torch.tensor(min_values, dtype=torch.float32))
        self.register_buffer("max_values", torch.tensor(max_values, dtype=torch.float32))

    def _prepare_re(self, re, batch_size, device):
        if re is None:
            if self.default_re is not None:
                re = torch.full((batch_size,), float(self.default_re), device=device)
            else:
                # Fallback keeps old call sites alive; normal training/validation passes Re.
                re = torch.full((batch_size,), float(self.re_values[0].item()), device=device)
        elif not torch.is_tensor(re):
            re = torch.tensor(re, dtype=torch.float32, device=device)
        else:
            re = re.to(device=device, dtype=torch.float32)

        if re.ndim == 0:
            re = re.repeat(batch_size)
        elif re.numel() == 1 and batch_size > 1:
            re = re.reshape(1).repeat(batch_size)
        else:
            re = re.reshape(-1)
        if re.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} Re values, got {re.numel()}")
        return re.clamp_min(self.eps)

    def _stats_for_re(self, re, batch_size, device):
        re = self._prepare_re(re, batch_size, device)
        log_re = torch.log(re)
        log_table = self.log_re_values.to(device)
        min_table = self.min_values.to(device)
        max_table = self.max_values.to(device)

        if log_table.numel() == 1:
            return min_table[0].expand(batch_size, -1), max_table[0].expand(batch_size, -1)

        hi = torch.searchsorted(log_table, log_re, right=False)
        hi = hi.clamp(1, log_table.numel() - 1)
        lo = hi - 1

        lo_log = log_table[lo]
        hi_log = log_table[hi]
        weight = (log_re - lo_log) / (hi_log - lo_log + self.eps)
        weight = weight.unsqueeze(-1)

        min_val = min_table[lo] * (1.0 - weight) + min_table[hi] * weight
        max_val = max_table[lo] * (1.0 - weight) + max_table[hi] * weight
        return min_val, max_val

    def _view_stats(self, values, x):
        shape = [values.shape[0], values.shape[1]] + [1] * (x.ndim - 2)
        return values.view(*shape)

    def normalize(self, x, re=None):
        min_val, max_val = self._stats_for_re(re, x.shape[0], x.device)
        min_val = self._view_stats(min_val, x)
        max_val = self._view_stats(max_val, x)
        x_std = (x - min_val) / (max_val - min_val + self.eps)
        return x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def denormalize(self, x, re=None):
        min_val, max_val = self._stats_for_re(re, x.shape[0], x.device)
        min_val = self._view_stats(min_val, x)
        max_val = self._view_stats(max_val, x)
        x_std = (x - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        return x_std * (max_val - min_val + self.eps) + min_val

    def to(self, device):
        self.re_values = self.re_values.to(device)
        self.log_re_values = self.log_re_values.to(device)
        self.min_values = self.min_values.to(device)
        self.max_values = self.max_values.to(device)
        return self


@register_normalization("minmax")
def create_minmax_normalization(norm_params):
    """Create a min-max normalization."""
    return MinMaxNormalization(
        min_val=norm_params.get("min_val"),
        max_val=norm_params.get("max_val"),
        feature_range=norm_params.get("feature_range", (0, 1)),
        eps=norm_params.get("eps", 1e-8),
    )

@register_normalization("paired_minmax")
def create_paired_minmax_normalization(norm_params):
    """Create a min-max normalization with shared vector-component scales."""
    return PairedMinMaxNormalization(
        min_val=norm_params.get("min_val"),
        max_val=norm_params.get("max_val"),
        feature_range=norm_params.get("feature_range", (0, 1)),
        channel_groups=norm_params.get("channel_groups", ((0, 1), (2, 3))),
        eps=norm_params.get("eps", 1e-8),
    )



@register_normalization("per_re_paired_minmax")
def create_per_re_paired_minmax_normalization(norm_params):
    """Create Re-dependent paired min-max normalization."""
    return PerRePairedMinMaxNormalization(
        stats_by_re=norm_params.get("stats_by_re"),
        feature_range=norm_params.get("feature_range", (0, 1)),
        channel_groups=norm_params.get("channel_groups", ((0, 1), (2, 3))),
        default_re=norm_params.get("default_re"),
        eps=norm_params.get("eps", 1e-8),
    )
