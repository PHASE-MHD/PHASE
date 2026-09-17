"""Training loop for the previous-study full-field DINO baseline."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from phase.data import get_diffusion_dataloaders
from phase.diffusion import create_diffusion_model
from phase.losses import LpLoss
from phase.optimizers import create_optimizer, create_scheduler
from phase.utils.diffusion_tensor_normalization import denormalize_with_channel_mapping


def _train_epoch(model, loader, optimizer, device, clip_norm):
    model.train()
    total = 0.0
    for condition, target in loader:
        condition = condition.to(device, non_blocking=True).contiguous()
        target = target.to(device, non_blocking=True).contiguous()
        optimizer.zero_grad(set_to_none=True)
        loss = model(target, condition)
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total += float(loss.detach())
    return total / max(len(loader), 1)


def _validate(model, loader, device, num_sample_steps):
    model.eval()
    normalizer = loader.dataset.target_normalizer
    channels = model.channels
    channel_indices = list(range(channels))
    relative_l2 = LpLoss()
    mse = nn.MSELoss()
    totals = [0.0, 0.0, 0.0]
    with torch.no_grad():
        for condition, target in loader:
            condition = condition.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            prediction = model.sample(condition, num_sample_steps=num_sample_steps)
            totals[0] += float(mse(prediction, target))
            if normalizer is not None:
                prediction = denormalize_with_channel_mapping(
                    prediction, normalizer, channel_indices
                )
                target = denormalize_with_channel_mapping(
                    target, normalizer, channel_indices
                )
            totals[1] += float(relative_l2(prediction, target))
            totals[2] += float(mse(prediction, target))
    count = max(len(loader), 1)
    return tuple(value / count for value in totals)


def _checkpoint(path, model, optimizer, scheduler, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": metrics[0],
            "denorm_loss_rel_l2": metrics[1],
            "denorm_loss_mse": metrics[2],
        },
        path,
    )


def train_dino(config):
    """Train full-field DINO from scratch and return its metric history."""
    dataset = config["dataset_params"]
    if dataset.get("prediction_mode") != "direct" or dataset.get("residual_target"):
        raise ValueError("The previous DINO baseline requires direct full-field targets.")
    model_params = config["model_params"]
    forbidden = (
        model_params.get("helmholtz_projection", False),
        model_params.get("use_vorticity_loss", False),
        model_params.get("use_current_loss", False),
        model_params.get("re_conditioning", {}).get("enabled", False),
    )
    if any(forbidden):
        raise ValueError("The previous DINO baseline disables PHASE physics additions.")
    train = config["train_params"]
    if str(train.get("load_checkpoint", "")).strip():
        raise ValueError("The previous DINO diffusion model must start from scratch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = config["dataloader_params"]["train"].get("num_workers", 1)
    train_loader, val_loader, test_loader = get_diffusion_dataloaders(
        config, num_workers=workers
    )
    model = create_diffusion_model(config).to(device)
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
        if epoch % interval == 0 or epoch == int(train["epochs"]) - 1:
            metrics = _validate(model, val_loader, device, sample_steps)
            row.update(
                validation_mse=metrics[0],
                validation_denorm_rel_l2=metrics[1],
                validation_denorm_mse=metrics[2],
            )
            if metrics[1] < best:
                best = metrics[1]
                _checkpoint(
                    train["checkpoint_path"], model, optimizer, scheduler,
                    epoch, metrics,
                )
        history.append(row)
        print(row, flush=True)

    saved = torch.load(train["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    test_metrics = _validate(model, test_loader, device, sample_steps)
    result = {
        "checkpoint_epoch": saved["epoch"],
        "test_mse": test_metrics[0],
        "test_denorm_rel_l2": test_metrics[1],
        "test_denorm_mse": test_metrics[2],
        "history": history,
    }
    print(result, flush=True)
    return result
