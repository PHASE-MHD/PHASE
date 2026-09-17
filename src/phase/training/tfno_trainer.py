"""Training loop for the previous-study tFNO baseline."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn

from phase.data import get_dataloaders
from phase.losses import LpLoss, create_loss
from phase.models import create_model
from phase.optimizers import create_optimizer, create_scheduler


def _criterion_loss(criterion, loader, prediction, target, inputs):
    if getattr(criterion, "requires_input_data", False):
        return criterion(loader, prediction, target, inputs)
    return criterion(prediction, target)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True).contiguous()
            target = target.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = _criterion_loss(criterion, loader, prediction, target, inputs)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += float(loss.detach())
    return total / max(len(loader), 1)


def _denormalized_metrics(model, loader, device):
    model.eval()
    normalizer = copy.deepcopy(getattr(loader.dataset, "normalizer", None))
    relative_l2 = LpLoss()
    mse = nn.MSELoss()
    totals = [0.0, 0.0]
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True).contiguous()
            target = target.to(device, non_blocking=True)
            prediction = model(inputs)
            if normalizer is not None:
                normalizer = normalizer.to(device)
                prediction = normalizer.denormalize(prediction)
                target = normalizer.denormalize(target)
            totals[0] += float(relative_l2(prediction, target))
            totals[1] += float(mse(prediction, target))
    count = max(len(loader), 1)
    return totals[0] / count, totals[1] / count


def _save_checkpoint(path, model, optimizer, scheduler, epoch, val_loss, rel_l2, mse):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": val_loss,
            "denorm_loss_rel_l2": rel_l2,
            "denorm_loss_mse": mse,
        },
        path,
    )


def train_tfno(config):
    """Train the configured tFNO and return final test metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = config["dataset_params"]
    loaders = config["dataloader_params"]
    normalization = {"normalization_params": config["normalization_params"]}
    train_loader, val_loader, test_loader = get_dataloaders(
        dataset["data_path"],
        normalization_config=normalization,
        batch_size=loaders["train"]["batch_size"],
        num_workers=loaders["train"].get("num_workers", 0),
        train_size=dataset["train_size"],
        val_plus_test_size=dataset["val_plus_test_size"],
        seed=dataset.get("seed", 42),
        sub_t=dataset.get("sub_t", 1),
        sub_x=dataset.get("sub_x", 1),
        t_range=tuple(dataset.get("t_range", (0.0, 1.0))),
        x_range=tuple(dataset.get("x_range", (0.0, 1.0))),
        y_range=tuple(dataset.get("y_range", (0.0, 1.0))),
    )

    model = create_model(config).to(device)
    criterion = create_loss(config)
    optimizer = create_optimizer(model.parameters(), config)
    scheduler = create_scheduler(optimizer, config)
    train = config["train_params"]
    load_checkpoint = str(train.get("load_checkpoint", "")).strip()
    if load_checkpoint:
        raise ValueError(
            "The canonical Batch 4 tFNO is a no-warm-start baseline; "
            "load_checkpoint must remain empty."
        )

    best = math.inf
    history = []
    for epoch in range(train["epochs"]):
        train_loss = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = _run_epoch(model, val_loader, criterion, device)
        val_rel_l2, val_mse = _denormalized_metrics(model, val_loader, device)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_denorm_rel_l2": val_rel_l2,
            "val_denorm_mse": val_mse,
        }
        history.append(row)
        print(row, flush=True)
        if val_loss < best:
            best = val_loss
            _save_checkpoint(
                train["checkpoint_path"], model, optimizer, scheduler, epoch,
                val_loss, val_rel_l2, val_mse,
            )

    checkpoint = torch.load(train["checkpoint_path"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss = _run_epoch(model, test_loader, criterion, device)
    test_rel_l2, test_mse = _denormalized_metrics(model, test_loader, device)
    result = {
        "checkpoint_epoch": checkpoint["epoch"],
        "test_loss": test_loss,
        "test_denorm_rel_l2": test_rel_l2,
        "test_denorm_mse": test_mse,
        "history": history,
    }
    print(result, flush=True)
    return result
