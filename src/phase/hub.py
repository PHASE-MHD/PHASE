"""Load complete PHASE and DINO pipelines from local or Hugging Face packages."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from phase.data.diffusion_dataset import _load_per_re_stats
from phase.diffusion import create_diffusion_model
from phase.models import create_model, map_official_tfno_state_dict
from phase.normalizations import create_normalization
from phase.utils.batches import model_forward
from phase.utils.diffusion_tensor_normalization import (
    denormalize_with_channel_mapping,
    normalize_with_channel_mapping,
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")
    return value


def _expand_model_root(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _expand_model_root(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_model_root(item, root) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value.replace("${MODEL_ROOT}", str(root)))
    return value


def _torch_load(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = _torch_load(path)
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint {path} does not contain a state dictionary.")
    return checkpoint


def _normalizer_from_params(params: Mapping[str, Any], kind: str | None = None):
    params = copy.deepcopy(dict(params))
    norm_type = str(params.get("type", "identity")).lower()
    if norm_type == "per_re_paired_minmax":
        if kind not in {"inputs", "targets"}:
            raise ValueError("Per-Re normalization requires inputs or targets kind.")
        params["stats_by_re"] = _load_per_re_stats(params, kind)
    else:
        stats_path = params.get(f"{kind}_stats_file") if kind else None
        stats_path = stats_path or params.get("stats_file")
        if stats_path:
            with np.load(stats_path) as stats:
                for key in stats.files:
                    value = stats[key]
                    params[key] = value.tolist() if hasattr(value, "tolist") else value
    return create_normalization({"normalization_params": params})


class PHASEPipeline:
    """A deterministic conditioner followed by full-field or residual diffusion."""

    def __init__(
        self,
        conditioner,
        diffusion,
        conditioner_config: Mapping[str, Any],
        diffusion_config: Mapping[str, Any],
        conditioner_normalizer,
        input_normalizer,
        target_normalizer,
        *,
        model_id: str,
    ):
        self.conditioner = conditioner
        self.diffusion = diffusion
        self.conditioner_config = dict(conditioner_config)
        self.diffusion_config = dict(diffusion_config)
        self.conditioner_normalizer = conditioner_normalizer
        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer
        self.model_id = model_id
        self.residual_target = bool(
            self.diffusion_config.get("dataset_params", {}).get("residual_target", False)
        )
        self.channels = int(self.diffusion_config["model_params"]["channels"])
        self.channel_indices = list(range(self.channels))

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str | os.PathLike,
        *,
        revision: str | None = None,
        cache_dir: str | os.PathLike | None = None,
        token: str | bool | None = None,
        device: str | torch.device = "cpu",
    ) -> "PHASEPipeline":
        """Load a complete published pipeline by Hugging Face URI or local path."""
        source = Path(model_id_or_path)
        if source.is_dir():
            root = source.resolve()
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise ImportError(
                    "Install PHASE with `pip install 'phase[hub]'` to load a Hub URI."
                ) from exc
            root = Path(
                snapshot_download(
                    repo_id=str(model_id_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                )
            )

        manifest = _load_yaml(root / "pipeline.yaml")
        if int(manifest.get("format_version", 0)) != 1:
            raise ValueError("Unsupported PHASE pipeline package format.")
        conditioner_spec = manifest["conditioner"]
        diffusion_spec = manifest["diffusion"]
        conditioner_config = _expand_model_root(
            _load_yaml(root / conditioner_spec["config"]), root
        )
        diffusion_config = _expand_model_root(
            _load_yaml(root / diffusion_spec["config"]), root
        )

        conditioner = create_model(conditioner_config)
        conditioner_state = _state_dict(root / conditioner_spec["weights"])
        if conditioner_spec.get("official_tfno_mapping", False):
            conditioner_state = map_official_tfno_state_dict(conditioner_state)
        conditioner.load_state_dict(conditioner_state, strict=True)

        diffusion = create_diffusion_model(diffusion_config)
        diffusion.load_state_dict(
            _state_dict(root / diffusion_spec["weights"]), strict=True
        )

        conditioner_normalizer = _normalizer_from_params(
            conditioner_config.get("normalization_params", {"type": "identity"})
        )
        diffusion_norm = diffusion_config.get(
            "normalization_params", {"type": "identity"}
        )
        input_normalizer = _normalizer_from_params(diffusion_norm, "inputs")
        target_normalizer = _normalizer_from_params(diffusion_norm, "targets")

        pipeline = cls(
            conditioner,
            diffusion,
            conditioner_config,
            diffusion_config,
            conditioner_normalizer,
            input_normalizer,
            target_normalizer,
            model_id=str(model_id_or_path),
        )
        return pipeline.to(device).eval()

    def to(self, device: str | torch.device) -> "PHASEPipeline":
        self.device = torch.device(device)
        self.conditioner.to(self.device)
        self.diffusion.to(self.device)
        for normalizer in (
            self.conditioner_normalizer,
            self.input_normalizer,
            self.target_normalizer,
        ):
            if hasattr(normalizer, "to"):
                normalizer.to(self.device)
        return self

    def eval(self) -> "PHASEPipeline":
        self.conditioner.eval()
        self.diffusion.eval()
        return self

    def _metadata(self, batch_size: int, re: float | torch.Tensor | None):
        if re is None:
            return None
        values = torch.as_tensor(re, dtype=torch.float32, device=self.device).reshape(-1)
        if values.numel() == 1:
            values = values.repeat(batch_size)
        if values.numel() != batch_size:
            raise ValueError("Re must be scalar or contain one value per batch item.")
        return {"re": values, "rem": values}

    def _normalize_conditioner_input(self, fields, metadata):
        try:
            return self.conditioner_normalizer.normalize(fields, metadata=metadata)
        except TypeError:
            return self.conditioner_normalizer.normalize(fields)

    def condition(
        self,
        initial_fields: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        *,
        re: float | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict physical trajectories from physical initial fields."""
        fields = torch.as_tensor(initial_fields, dtype=torch.float32, device=self.device)
        if fields.ndim != 4 or fields.shape[1] != self.channels:
            raise ValueError(
                f"Expected initial_fields [B,{self.channels},H,W], got {tuple(fields.shape)}."
            )
        times = torch.as_tensor(times, dtype=fields.dtype, device=self.device)
        if times.ndim != 1:
            raise ValueError("times must be a one-dimensional sequence.")
        modes = getattr(self.conditioner, "num_fno_modes", None)
        if isinstance(modes, int) and len(times) < modes:
            raise ValueError(
                f"The tFNO conditioner requires at least {modes} time points; "
                f"received {len(times)}. Use the trained 26-frame grid for this model."
            )
        batch, channels, height, width = fields.shape
        metadata = self._metadata(batch, re)
        repeated = fields.unsqueeze(2).expand(batch, channels, len(times), height, width)
        normalized = self._normalize_conditioner_input(repeated, metadata)

        t_grid = times.view(1, 1, -1, 1, 1).expand(batch, 1, -1, height, width)
        x = torch.linspace(0, 1, height, device=self.device, dtype=fields.dtype)
        y = torch.linspace(0, 1, width, device=self.device, dtype=fields.dtype)
        x_grid = x.view(1, 1, 1, height, 1).expand(batch, 1, len(times), height, width)
        y_grid = y.view(1, 1, 1, 1, width).expand(batch, 1, len(times), height, width)
        model_input = torch.cat((t_grid, x_grid, y_grid, normalized), dim=1)
        with torch.no_grad():
            prediction = model_forward(self.conditioner, model_input, metadata)
        try:
            return self.conditioner_normalizer.denormalize(
                prediction, metadata=metadata
            )
        except TypeError:
            return self.conditioner_normalizer.denormalize(prediction)

    def predict(
        self,
        initial_fields: torch.Tensor,
        times: Sequence[float] | torch.Tensor,
        *,
        re: float | torch.Tensor | None = None,
        seed: int | None = None,
        num_sample_steps: int | None = None,
        diffusion_batch_size: int = 1,
    ) -> torch.Tensor:
        """Run conditioner and diffusion; return physical fields [B,C,T,H,W]."""
        condition = self.condition(initial_fields, times, re=re)
        batch, channels, steps, height, width = condition.shape
        flattened = condition.permute(0, 2, 1, 3, 4).reshape(
            batch * steps, channels, height, width
        )
        re_values = None
        if re is not None:
            re_values = self._metadata(batch, re)["re"].repeat_interleave(steps)
        normalized_condition = normalize_with_channel_mapping(
            flattened,
            self.input_normalizer,
            self.channel_indices,
            re=re_values,
        )
        if seed is not None:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        if diffusion_batch_size < 1:
            raise ValueError("diffusion_batch_size must be at least 1.")
        chunks = []
        with torch.no_grad():
            for start in range(0, len(normalized_condition), diffusion_batch_size):
                chunks.append(
                    self.diffusion.sample(
                        normalized_condition[start : start + diffusion_batch_size],
                        num_sample_steps=num_sample_steps,
                        apply_projection=not self.residual_target,
                    )
                )
        generated = torch.cat(chunks, dim=0)
        generated = denormalize_with_channel_mapping(
            generated,
            self.target_normalizer,
            self.channel_indices,
            re=re_values,
        )
        if self.residual_target:
            generated = flattened + generated
            if getattr(self.diffusion, "projection_module", None) is not None:
                generated = self.diffusion.projection_module(generated)
        return generated.reshape(batch, steps, channels, height, width).permute(
            0, 2, 1, 3, 4
        )
