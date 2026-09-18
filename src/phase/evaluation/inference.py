"""Test-only inference adapters for all public PHASE model families."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterator

import torch

from phase.data import (
    get_dataloaders,
    get_diffusion_test_dataloader,
    get_multi_re_dataloaders,
)
from phase.diffusion import create_diffusion_model
from phase.models import create_model
from phase.utils.batches import model_forward, move_metadata_to_device, unpack_batch
from phase.utils.diffusion_tensor_normalization import (
    denormalize_with_channel_mapping,
    reconstruct_residual_prediction,
)

from .metrics import EvaluationRecord
from .reporting import sha256_file


def model_family(config: dict) -> str:
    if config.get("config_type") == "diffusion":
        recipe = config.get("train_params", {}).get("recipe", "previous_dino")
        return "DINO" if recipe == "previous_dino" else "PHASE"
    model_type = str(config.get("model_params", {}).get("model_type", "")).lower()
    return "tFNO" if model_type == "tfno" else "scOT"


def representation(config: dict) -> str:
    model = config.get("model_params", {})
    channels = int(model.get("channels", model.get("out_channels", 0)))
    if channels == 3:
        return "vector_potential"
    if channels == 4:
        return "direct_b"
    raise ValueError(f"Evaluation requires three or four output channels, got {channels}.")


def _strict_load(
    model,
    checkpoint_path: str | Path,
    *,
    allow_official_tfno_mapping: bool = False,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        if not allow_official_tfno_mapping:
            raise
        from phase.models import map_official_tfno_state_dict

        model.load_state_dict(map_official_tfno_state_dict(state), strict=True)
    return checkpoint if isinstance(checkpoint, dict) else {}


def _denormalize(normalizer, value, metadata=None):
    if normalizer is None:
        return value
    normalizer = copy.deepcopy(normalizer).to(value.device)
    try:
        return normalizer.denormalize(value, metadata=metadata)
    except TypeError:
        return normalizer.denormalize(value)


def _deterministic_test_loader(config: dict, num_workers: int):
    dataset = config["dataset_params"]
    normalization = {"normalization_params": config["normalization_params"]}
    if dataset.get("dataset_type") == "multi_re" or dataset.get("re_values"):
        return get_multi_re_dataloaders(
            dataset,
            normalization_config=normalization,
            batch_size=1,
            num_workers=num_workers,
            seed=int(dataset.get("seed", 42)),
        )[2]
    return get_dataloaders(
        dataset["data_path"],
        normalization_config=normalization,
        batch_size=1,
        num_workers=num_workers,
        train_size=int(dataset["train_size"]),
        val_plus_test_size=int(dataset["val_plus_test_size"]),
        seed=int(dataset.get("seed", 42)),
        sub_t=int(dataset.get("sub_t", 1)),
        sub_x=int(dataset.get("sub_x", 1)),
        t_range=tuple(dataset.get("t_range", (0.0, 1.0))),
        x_range=tuple(dataset.get("x_range", (0.0, 1.0))),
        y_range=tuple(dataset.get("y_range", (0.0, 1.0))),
    )[2]


def _deterministic_records(
    config: dict,
    checkpoint_path: str | Path,
    *,
    target_re: float,
    device: torch.device,
    num_workers: int,
    max_samples: int | None,
) -> tuple[Iterator[EvaluationRecord], dict]:
    loader = _deterministic_test_loader(config, num_workers)
    model = create_model(config).to(device)
    checkpoint = _strict_load(
        model,
        checkpoint_path,
        allow_official_tfno_mapping=(model_family(config) == "tFNO"),
    )
    model.eval()
    normalizer = getattr(loader.dataset, "normalizer", None)
    output_representation = representation(config)

    def iterator():
        count = 0
        with torch.no_grad():
            for batch_index, batch in enumerate(loader):
                inputs, truth, metadata = unpack_batch(batch)
                metadata = move_metadata_to_device(metadata, device)
                if metadata is not None:
                    batch_re = float(metadata["re"].reshape(-1)[0])
                    if abs(batch_re - target_re) > 1.0e-6:
                        continue
                    sample_id = int(metadata["sample_index"].reshape(-1)[0])
                else:
                    batch_re = target_re
                    sample_id = int(loader.dataset.indices[batch_index])
                inputs = inputs.to(device, non_blocking=True).contiguous()
                truth = truth.to(device, non_blocking=True)
                prediction = model_forward(model, inputs, metadata)
                prediction = _denormalize(normalizer, prediction, metadata)
                truth = _denormalize(normalizer, truth, metadata)
                yield EvaluationRecord(
                    prediction=prediction[0].detach().cpu(),
                    truth=truth[0].detach().cpu(),
                    representation=output_representation,
                    sample_id=sample_id,
                    re=batch_re,
                )
                count += 1
                if max_samples is not None and count >= max_samples:
                    break

    return iterator(), {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "representation": output_representation,
        "sample_id_source": "source_dataset_index",
    }


def _constant_metadata_re(metadata, target_re: float) -> tuple[float, int | None]:
    if metadata is None:
        return target_re, None
    re_values = metadata.get("re")
    if re_values is None:
        re_value = target_re
    else:
        if not torch.allclose(re_values, re_values.reshape(-1)[0]):
            raise ValueError("A diffusion evaluation batch crossed Re trajectories.")
        re_value = float(re_values.reshape(-1)[0])
    sample_values = metadata.get("sample_id")
    sample_id = None
    if sample_values is not None:
        if not torch.equal(sample_values, sample_values.reshape(-1)[0].expand_as(sample_values)):
            raise ValueError("A diffusion evaluation batch crossed sample IDs.")
        sample_id = int(sample_values.reshape(-1)[0])
    return re_value, sample_id


def _diffusion_records(
    config: dict,
    checkpoint_path: str | Path,
    *,
    target_re: float,
    device: torch.device,
    num_workers: int,
    max_samples: int | None,
    diffusion_seed: int,
    num_sample_steps: int,
) -> tuple[Iterator[EvaluationRecord], dict]:
    configured_sigma_data = config.get("model_params", {}).get("sigma_data")
    test_loader = get_diffusion_test_dataloader(config, num_workers=num_workers)
    if configured_sigma_data is not None:
        config["model_params"]["sigma_data"] = configured_sigma_data
    model = create_diffusion_model(config).to(device)
    checkpoint = _strict_load(model, checkpoint_path)
    model.eval()
    dataset = test_loader.dataset
    residual_mode = bool(dataset.residual_target)
    channel_indices = dataset.channel_indices or list(range(model.channels))
    output_representation = representation(config)
    has_feature_sample_ids = getattr(dataset, "sample_id_per_sample", None) is not None

    def iterator():
        count = 0
        with torch.no_grad():
            for trajectory_index, batch in enumerate(test_loader):
                condition, target, metadata = unpack_batch(batch)
                metadata = move_metadata_to_device(metadata, device)
                batch_re, sample_id = _constant_metadata_re(metadata, target_re)
                if abs(batch_re - target_re) > 1.0e-6:
                    continue
                if sample_id is None:
                    ids = getattr(dataset, "sample_id_per_sample", None)
                    sample_id = (
                        int(ids[trajectory_index]) if ids is not None else trajectory_index
                    )
                torch.manual_seed(diffusion_seed + int(sample_id))
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(diffusion_seed + int(sample_id))
                condition = condition.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                prediction = model.sample(
                    condition,
                    num_sample_steps=num_sample_steps,
                    re=metadata.get("re") if metadata is not None else None,
                    rem=metadata.get("rem") if metadata is not None else None,
                )
                re_tensor = metadata.get("re") if metadata is not None else None
                if residual_mode:
                    prediction, target, _ = reconstruct_residual_prediction(
                        condition,
                        prediction,
                        target,
                        input_normalizer=dataset.input_normalizer,
                        target_normalizer=dataset.target_normalizer,
                        channel_indices=channel_indices,
                        re=re_tensor,
                        denormalize=True,
                        projection_module=getattr(model, "projection_module", None),
                        project_full_field=(
                            getattr(model, "projection_mode", "")
                            == "full_field_residual"
                        ),
                    )
                else:
                    prediction = denormalize_with_channel_mapping(
                        prediction,
                        dataset.target_normalizer,
                        channel_indices,
                        re=re_tensor,
                    )
                    target = denormalize_with_channel_mapping(
                        target,
                        dataset.target_normalizer,
                        channel_indices,
                        re=re_tensor,
                    )
                yield EvaluationRecord(
                    prediction=prediction.permute(1, 0, 2, 3).detach().cpu(),
                    truth=target.permute(1, 0, 2, 3).detach().cpu(),
                    representation=output_representation,
                    sample_id=int(sample_id),
                    re=batch_re,
                )
                count += 1
                if max_samples is not None and count >= max_samples:
                    break

    return iterator(), {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "representation": output_representation,
        "diffusion_seed": diffusion_seed,
        "num_sample_steps": num_sample_steps,
        "sample_id_source": (
            "feature_metadata" if has_feature_sample_ids else "test_split_position"
        ),
    }


def build_test_inference(
    config: dict,
    checkpoint_path: str | Path,
    *,
    target_re: float,
    device: torch.device,
    num_workers: int = 0,
    max_samples: int | None = None,
    diffusion_seed: int = 42,
    num_sample_steps: int = 32,
) -> tuple[Iterator[EvaluationRecord], dict]:
    """Build a held-out-test iterator and immutable run provenance."""
    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    family = model_family(config)
    if config.get("config_type") == "diffusion":
        records, details = _diffusion_records(
            config,
            checkpoint_path,
            target_re=target_re,
            device=device,
            num_workers=num_workers,
            max_samples=max_samples,
            diffusion_seed=diffusion_seed,
            num_sample_steps=num_sample_steps,
        )
    else:
        records, details = _deterministic_records(
            config,
            checkpoint_path,
            target_re=target_re,
            device=device,
            num_workers=num_workers,
            max_samples=max_samples,
        )
    return records, {
        "model": family,
        "split": "test",
        "re": float(target_re),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        **details,
    }
