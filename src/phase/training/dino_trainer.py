"""Training loop for full-field DINO and residual PHASE diffusion recipes."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from phase.data import get_diffusion_dataloaders
from phase.diffusion import create_diffusion_model
from phase.losses import LpLoss
from phase.optimizers import create_optimizer, create_scheduler
from phase.utils.batches import move_metadata_to_device, unpack_batch
from phase.utils.diffusion_tensor_normalization import (
    denormalize_with_channel_mapping,
    reconstruct_residual_prediction,
)


_SINGLE_RE_RESIDUAL_RECIPES = {
    "phase_residual_single_re",
    "kh_phase_residual_single_re",
}
_MULTI_RE_RESIDUAL_RECIPES = {
    "phase_residual_multi_re",
    "kh_phase_residual_multi_re",
}
_KH_RESIDUAL_RECIPES = {
    "kh_phase_residual_single_re",
    "kh_phase_residual_multi_re",
}
_RESIDUAL_RECIPES = _SINGLE_RE_RESIDUAL_RECIPES | _MULTI_RE_RESIDUAL_RECIPES


def _metadata_re(metadata):
    return metadata.get("re") if metadata is not None else None


def _should_validate(epoch, num_epochs, interval, recipe):
    """Match historical PHASE validation epochs without changing DINO."""
    if interval <= 0:
        raise ValueError("validation_interval must be positive.")
    if recipe in _RESIDUAL_RECIPES:
        return epoch != 0 and epoch % interval == 0
    return epoch % interval == 0 or epoch == num_epochs - 1


def _train_epoch(model, loader, optimizer, device, clip_norm):
    model.train()
    total = 0.0
    dataset = loader.dataset
    channel_indices = dataset.channel_indices or list(range(model.channels))
    for batch in loader:
        condition, target, metadata = unpack_batch(batch)
        condition = condition.to(device, non_blocking=True).contiguous()
        target = target.to(device, non_blocking=True).contiguous()
        metadata = move_metadata_to_device(metadata, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(
            target,
            condition,
            re=_metadata_re(metadata),
            rem=metadata.get("rem") if metadata is not None else None,
            input_normalizer=dataset.input_normalizer,
            target_normalizer=dataset.target_normalizer,
            channel_indices=channel_indices,
        )
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)


def _validate(model, loader, device, num_sample_steps):
    model.eval()
    dataset = loader.dataset
    normalizer = dataset.target_normalizer
    input_normalizer = dataset.input_normalizer
    residual_mode = dataset.residual_target
    channel_indices = dataset.channel_indices or list(range(model.channels))
    relative_l2 = LpLoss()
    mse = nn.MSELoss()
    totals = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for batch in loader:
            condition, target, metadata = unpack_batch(batch)
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            metadata = move_metadata_to_device(metadata, device)
            re = _metadata_re(metadata)
            prediction = model.sample(
                condition,
                num_sample_steps=num_sample_steps,
                re=re,
                rem=metadata.get("rem") if metadata is not None else None,
            )
            totals[0] += float(mse(prediction, target))
            if residual_mode:
                prediction, target, _ = reconstruct_residual_prediction(
                    condition,
                    prediction,
                    target,
                    input_normalizer=input_normalizer,
                    target_normalizer=normalizer,
                    channel_indices=channel_indices,
                    re=re,
                    denormalize=True,
                    projection_module=getattr(model, "projection_module", None),
                    project_full_field=(
                        getattr(model, "projection_mode", "")
                        == "full_field_residual"
                    ),
                )
            elif normalizer is not None:
                prediction = denormalize_with_channel_mapping(
                    prediction, normalizer, channel_indices, re=re
                )
                target = denormalize_with_channel_mapping(
                    target, normalizer, channel_indices, re=re
                )
            totals[1] += float(relative_l2(prediction, target))
            totals[2] += float(mse(prediction, target))
    count = max(len(loader), 1)
    return tuple(value / count for value in totals)


def _checkpoint(
    path, model, optimizer, scheduler, epoch, metrics, selected_metric_index=0
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": metrics[selected_metric_index],
            "model_val_loss": metrics[0],
            "denorm_loss_rel_l2": metrics[1],
            "denorm_loss_mse": metrics[2],
        },
        path,
    )


def _validate_recipe(config):
    """Lock baseline and PHASE diffusion recipes to audited semantics."""
    dataset = config["dataset_params"]
    model_params = config["model_params"]
    train = config["train_params"]
    recipe = train.get("recipe", "previous_dino")
    if recipe == "previous_dino":
        if dataset.get("prediction_mode") != "direct" or dataset.get("residual_target"):
            raise ValueError("The previous DINO baseline requires direct full-field targets.")
        forbidden = (
            model_params.get("helmholtz_projection", False),
            model_params.get("use_vorticity_loss", False),
            model_params.get("use_current_loss", False),
            model_params.get("re_conditioning", {}).get("enabled", False),
        )
        if any(forbidden):
            raise ValueError("The previous DINO baseline disables PHASE physics additions.")
        if str(train.get("load_checkpoint", "")).strip() or str(
            train.get("warm_start_checkpoint", "")
        ).strip():
            raise ValueError("The previous DINO diffusion model starts from scratch.")
        return recipe
    if recipe not in _RESIDUAL_RECIPES:
        raise ValueError("Unknown diffusion recipe: " + str(recipe))
    checks = {
        "no full-state resume": not str(train.get("load_checkpoint", "")).strip(),
        "denormalized relative-L2 checkpointing": (
            train.get("checkpoint_metric") == "denorm_rel_l2"
        ),
        "four output channels": model_params.get("channels") == 4,
        "residual targets": (
            dataset.get("prediction_mode") == "residual"
            and dataset.get("residual_target") is True
        ),
        "full-field Helmholtz projection": (
            model_params.get("helmholtz_projection") is True
            and model_params.get("project_velocity") is True
            and model_params.get("project_B") is True
            and model_params.get("projection_mode") == "full_field_residual"
        ),
        "no diffusion Re conditioning": not model_params.get(
            "re_conditioning", {}
        ).get("enabled", False),
        "no derivative-loss ablation": (
            not model_params.get("use_vorticity_loss", False)
            and not model_params.get("use_current_loss", False)
            and float(model_params.get("vorticity_loss_weight", 0.0)) == 0.0
            and float(model_params.get("current_loss_weight", 0.0)) == 0.0
        ),
        "positive validation interval": int(train.get("validation_interval", 0)) > 0,
        "single optimizer group": not config.get("optimizer_params", {})
        .get("param_groups", {})
        .get("enabled", False),
        "training length must match its declared mode": train.get("epochs")
        == (10 if train.get("acceptance_run") is True else 100),
    }
    if train.get("acceptance_run") is True:
        checks["acceptance source fractions"] = [
            dataset.get("source_train_sample_fraction"),
            dataset.get("source_validation_sample_fraction"),
            dataset.get("source_test_sample_fraction"),
        ] == [0.2, 0.1, 0.1]
    expected_norm = (
        "paired_minmax"
        if recipe in _SINGLE_RE_RESIDUAL_RECIPES
        else "per_re_paired_minmax"
    )
    checks[expected_norm + " normalization"] = (
        config.get("normalization_params", {}).get("type") == expected_norm
    )
    if recipe in _SINGLE_RE_RESIDUAL_RECIPES:
        checks["random diffusion initialization"] = not str(
            train.get("warm_start_checkpoint", "")
        ).strip()
    else:
        checks["weights-only historical warm start"] = (
            train.get("warm_start_mode") == "model_weights_only"
            and bool(str(train.get("warm_start_checkpoint", "")).strip())
        )
        checks["balanced all-Re batches"] = (
            dataset.get("balanced_re_batches") is True
            and dataset.get("res_per_batch")
            == len(config.get("normalization_params", {}).get("re_values", []))
        )
    if recipe in _KH_RESIDUAL_RECIPES:
        expected_model = {
            "model_type": "elucidated",
            "base_dim": 128,
            "dim_mults": [1, 2, 3, 5, 8, 12],
            "channels": 4,
            "self_condition": True,
            "learned_variance": False,
            "learned_sinusoidal_cond": False,
            "random_fourier_features": False,
            "sinusoidal_pos_emb_theta": 10000,
            "learned_sinusoidal_dim": 16,
            "dropout": 0.0,
            "attn_heads": 8,
            "attn_dim_head": 64,
            "flash_attn": True,
            "image_size": 128,
            "num_sample_steps": 32,
            "sigma_min": 0.002,
            "sigma_max": 80,
            "sigma_data": 0.5,
            "rho": 7,
            "P_mean": -1.2,
            "P_std": 1.2,
            "S_churn": 80,
            "S_tmin": 0.05,
            "S_tmax": 50,
            "S_noise": 1.003,
        }
        expected_optimizer = {
            "optimizer_type": "adamw",
            "lr": 5.0e-5,
            "weight_decay": 1.0e-4,
            "betas": [0.9, 0.999],
            "use_scheduler": True,
            "scheduler_type": "cosine",
            "T_max": 1000,
            "eta_min": 2.5e-5,
        }
        optimizer = config.get("optimizer_params", {})
        loader = config.get("train_loader_params", {})
        runtime_loader = config.get("dataloader_params", {}).get("train", {})
        normalization = config.get("normalization_params", {})
        expected_workers = (
            1 if recipe == "kh_phase_residual_single_re" else 8
        )
        checks.update(
            {
                "canonical KH EDM architecture and schedule": all(
                    model_params.get(key) == value
                    for key, value in expected_model.items()
                ),
                "canonical KH optimizer and scheduler": all(
                    optimizer.get(key) == value
                    for key, value in expected_optimizer.items()
                ),
                "canonical KH train loader": (
                    loader.get("batch_size") == 64
                    and loader.get("shuffle") is True
                    and loader.get("num_workers") == expected_workers
                    and loader.get("pin_memory") is True
                    and runtime_loader == loader
                ),
                "canonical KH paired channel groups": (
                    normalization.get("channel_groups") == [[0, 1], [2, 3]]
                    and normalization.get("feature_range") == [-1, 1]
                ),
                "KH source interval t=[0,5]": dataset.get("source_time_range")
                == [0.0, 5.0],
                "51 KH frames at physical dt=0.1": (
                    dataset.get("source_output_dt") == 0.02
                    and dataset.get("source_sub_t") == 5
                    and dataset.get("frames_per_trajectory") == 51
                ),
                "100-epoch KH training": train.get("epochs") == 100,
                "five-epoch KH full validation": train.get(
                    "validation_interval"
                )
                == 5,
            }
        )
        if recipe == "kh_phase_residual_single_re":
            checks["single-Re KH random-start mode"] = (
                train.get("warm_start_mode") == "none"
            )
        else:
            checks["canonical KH Re grid and metadata"] = (
                normalization.get("re_values")
                == [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]
                and normalization.get("default_re") == 1000.0
                and dataset.get("return_metadata") is True
            )
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError("Invalid " + recipe + " config: " + ", ".join(failed))
    return recipe


def _warm_start_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    source_epoch = checkpoint.get("epoch", "unknown")
    print(
        f"Warm-started model weights only from {checkpoint_path} "
        f"(source epoch={source_epoch}); epoch and optimizer reset.",
        flush=True,
    )


def _resume_training_state(
    model, optimizer, scheduler, checkpoint_path, metric_key="denorm_loss_rel_l2"
):
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    required = {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(
            f"Resume checkpoint {checkpoint_path} is missing: {missing}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = int(checkpoint["epoch"]) + 1
    best = float(
        checkpoint.get(metric_key, checkpoint.get("loss", math.inf))
    )
    print(
        f"Resumed full training state from {checkpoint_path} "
        f"at epoch={checkpoint['epoch']}; next epoch={start_epoch}.",
        flush=True,
    )
    return start_epoch, best


def train_dino(config, resume_checkpoint=None):
    """Train a locked full-field DINO or residual PHASE diffusion recipe."""
    recipe = _validate_recipe(config)
    train = config["train_params"]
    model_params = config["model_params"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = config["dataloader_params"]["train"].get("num_workers", 1)
    train_loader, val_loader, test_loader = get_diffusion_dataloaders(
        config, num_workers=workers
    )
    model = create_diffusion_model(config).to(device)
    warm_start = str(train.get("warm_start_checkpoint", "")).strip()
    if warm_start and not resume_checkpoint:
        _warm_start_weights(model, warm_start)
    parameters = model.get_optimizer_parameters(config["optimizer_params"])
    optimizer = create_optimizer(parameters, config)
    scheduler = create_scheduler(optimizer, config)
    start_epoch = 0
    best = math.inf
    if resume_checkpoint:
        start_epoch, best = _resume_training_state(
            model,
            optimizer,
            scheduler,
            resume_checkpoint,
            metric_key=(
                "denorm_loss_rel_l2"
                if recipe in _RESIDUAL_RECIPES
                else "model_val_loss"
            ),
        )
    selected_checkpoint_path = (
        resume_checkpoint if resume_checkpoint else train["checkpoint_path"]
    )
    clip_norm = (
        float(train.get("clip_grad_max_norm", 1.0))
        if train.get("clip_grad", True)
        else None
    )
    interval = int(train.get("validation_interval", 10))
    sample_steps = int(model_params.get("num_sample_steps", 32))
    history = []

    for epoch in range(start_epoch, int(train["epochs"])):
        train_loss = _train_epoch(model, train_loader, optimizer, device, clip_norm)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(train_loss)
        else:
            scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss}
        should_validate = _should_validate(
            epoch, int(train["epochs"]), interval, recipe
        )
        if should_validate:
            metrics = _validate(model, val_loader, device, sample_steps)
            row.update(
                validation_mse=metrics[0],
                validation_denorm_rel_l2=metrics[1],
                validation_denorm_mse=metrics[2],
            )
            if metrics[1] < best:
                best = metrics[1]
                _checkpoint(
                    train["checkpoint_path"],
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    metrics,
                    selected_metric_index=(
                        1 if recipe in _RESIDUAL_RECIPES else 0
                    ),
                )
                selected_checkpoint_path = train["checkpoint_path"]
        history.append(row)
        print(row, flush=True)

    saved = torch.load(
        selected_checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(saved["model_state_dict"], strict=True)
    test_metrics = _validate(model, test_loader, device, sample_steps)
    result = {
        "recipe": recipe,
        "checkpoint_epoch": saved["epoch"],
        "test_mse": test_metrics[0],
        "test_denorm_rel_l2": test_metrics[1],
        "test_denorm_mse": test_metrics[2],
        "history": history,
    }
    print(result, flush=True)
    return result
