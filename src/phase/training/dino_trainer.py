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


_RESIDUAL_RECIPES = {"phase_residual_single_re", "phase_residual_multi_re"}


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
    }
    expected_norm = (
        "paired_minmax"
        if recipe == "phase_residual_single_re"
        else "per_re_paired_minmax"
    )
    checks[expected_norm + " normalization"] = (
        config.get("normalization_params", {}).get("type") == expected_norm
    )
    if recipe == "phase_residual_single_re":
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


def train_dino(config):
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
    if warm_start:
        _warm_start_weights(model, warm_start)
    parameters = model.get_optimizer_parameters(config["optimizer_params"])
    optimizer = create_optimizer(parameters, config)
    scheduler = create_scheduler(optimizer, config)
    clip_norm = (
        float(train.get("clip_grad_max_norm", 1.0))
        if train.get("clip_grad", True)
        else None
    )
    interval = int(train.get("validation_interval", 10))
    sample_steps = int(model_params.get("num_sample_steps", 32))
    best = math.inf
    history = []

    for epoch in range(int(train["epochs"])):
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
        history.append(row)
        print(row, flush=True)

    saved = torch.load(train["checkpoint_path"], map_location=device, weights_only=False)
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
