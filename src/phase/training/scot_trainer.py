"""Training loop for the three-channel scOT ablations."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import torch
from torch import nn

from phase.data import get_dataloaders, get_multi_re_dataloaders
from phase.losses import LpLoss, create_loss
from phase.models import create_model
from phase.optimizers import create_optimizer, create_scheduler
from phase.utils.batches import model_forward, move_metadata_to_device, unpack_batch


def _criterion_loss(criterion, loader, prediction, target, inputs, metadata=None):
    if getattr(criterion, "requires_input_data", False):
        return criterion(loader, prediction, target, inputs, metadata=metadata)
    return criterion(prediction, target)


def _run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    component_totals = {}
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs, target, metadata = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True).contiguous()
            target = target.to(device, non_blocking=True)
            metadata = move_metadata_to_device(metadata, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            prediction = model_forward(model, inputs, metadata)
            loss = _criterion_loss(
                criterion, loader, prediction, target, inputs, metadata
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += float(loss.detach())
            for name, value in getattr(criterion, "last_components", {}).items():
                component_totals[name] = component_totals.get(name, 0.0) + float(value)
    count = max(len(loader), 1)
    components = {name: value / count for name, value in component_totals.items()}
    return total / count, components


def _denormalized_metrics(model, loader, device):
    model.eval()
    normalizer = copy.deepcopy(getattr(loader.dataset, "normalizer", None))
    relative_l2 = LpLoss()
    mse = nn.MSELoss()
    totals = [0.0, 0.0]
    with torch.no_grad():
        for batch in loader:
            inputs, target, metadata = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True).contiguous()
            target = target.to(device, non_blocking=True)
            metadata = move_metadata_to_device(metadata, device)
            prediction = model_forward(model, inputs, metadata)
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


def _validate_scratch_ablation(config):
    model = config["model_params"]
    train = config["train_params"]
    batch_sizes = {
        split: config["dataloader_params"][split]["batch_size"]
        for split in ("train", "validation", "test")
    }
    checks = {
        "load_pretrained_poseidon must be false": not model.get(
            "load_pretrained_poseidon", True
        ),
        "poseidon_model must be null": model.get("poseidon_model") is None,
        "POSEIDON-native fluid normalization must be disabled": not model.get(
            "use_poseidon_fluid_normalization", True
        ),
        "the locked no-transfer ablation uses batch_size=1 for every split": all(
            value == 1 for value in batch_sizes.values()
        ),
        "load_checkpoint must be empty": not str(
            train.get("load_checkpoint", "")
        ).strip(),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid scOT-without-TL config: " + "; ".join(failed))


def _validate_transfer_learning_ablation(config):
    model = config["model_params"]
    train = config["train_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer.get("param_groups", {})
    batch_sizes = {
        split: config["dataloader_params"][split]["batch_size"]
        for split in ("train", "validation", "test")
    }
    checks = {
        "load_pretrained_poseidon must be true": model.get(
            "load_pretrained_poseidon", False
        ),
        "poseidon_model must be camlab-ethz/Poseidon-T": model.get(
            "poseidon_model"
        )
        == "camlab-ethz/Poseidon-T",
        "POSEIDON-native fluid normalization must be enabled": model.get(
            "use_poseidon_fluid_normalization", False
        ),
        "velocity residual learning must be disabled": not model.get(
            "velocity_residual", False
        ),
        "magnetic residual learning must be enabled": model.get(
            "magnetic_residual", False
        ),
        "the locked transfer ablation uses batch_size=16 for every split": all(
            value == 16 for value in batch_sizes.values()
        ),
        "optimizer parameter groups must be enabled": groups.get("enabled", False),
        "pretrained_lr must be 5e-6": groups.get("pretrained_lr") == 5.0e-6,
        "new_lr must be 5e-4": groups.get("new_lr") == 5.0e-4,
        "pretrained_weight_decay must be 1e-2": groups.get(
            "pretrained_weight_decay"
        )
        == 1.0e-2,
        "new_weight_decay must be zero": groups.get("new_weight_decay") == 0.0,
        "load_checkpoint must be empty": not str(
            train.get("load_checkpoint", "")
        ).strip(),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid scOT-with-TL config: " + "; ".join(failed))


def _validate_naive_multi_re_ablation(config):
    model = config["model_params"]
    dataset = config["dataset_params"]
    train = config["train_params"]
    groups = config["optimizer_params"].get("param_groups", {})
    conditioning = model.get("re_input_conditioning", {})
    expected_re = [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]
    batch_sizes = {
        split: config["dataloader_params"][split]["batch_size"]
        for split in ("train", "validation", "test")
    }
    checks = {
        "model_type must be poseidon-mhd-re-input-finetune": model.get(
            "model_type"
        )
        == "poseidon-mhd-re-input-finetune",
        "load_pretrained_poseidon must be true": model.get(
            "load_pretrained_poseidon", False
        ),
        "POSEIDON-native fluid normalization must be enabled": model.get(
            "use_poseidon_fluid_normalization", False
        ),
        "velocity residual learning must be disabled": not model.get(
            "velocity_residual", False
        ),
        "magnetic residual learning must be enabled": model.get(
            "magnetic_residual", False
        ),
        "Re/Rm conditioning must include Rm": conditioning.get(
            "include_rem", False
        ),
        "Re/Rm input channels must use zero initialization": conditioning.get(
            "channel_init"
        )
        == "zero",
        "log_re_mean must be 2.9756": conditioning.get("log_re_mean") == 2.9756,
        "log_re_std must be 0.5417": conditioning.get("log_re_std") == 0.5417,
        "reference_re must be 1000": conditioning.get("reference_re") == 1000,
        "the locked ten Re values must be used": dataset.get("re_values")
        == expected_re,
        "Rm values must match Re values": dataset.get("rem_values") == expected_re,
        "balanced Re batches must be enabled": dataset.get(
            "balanced_re_batches", False
        ),
        "res_per_batch must be 10": dataset.get("res_per_batch") == 10,
        "the nominal batch_size must be 1 for every split": all(
            value == 1 for value in batch_sizes.values()
        ),
        "optimizer parameter groups must be enabled": groups.get("enabled", False),
        "pretrained_lr must be 1e-7": groups.get("pretrained_lr") == 1.0e-7,
        "new_lr must be 1e-3": groups.get("new_lr") == 1.0e-3,
        "pretrained_weight_decay must be 1e-2": groups.get(
            "pretrained_weight_decay"
        )
        == 1.0e-2,
        "new_weight_decay must be zero": groups.get("new_weight_decay") == 0.0,
        "warm_start_checkpoint must be set": bool(
            str(train.get("warm_start_checkpoint", "")).strip()
        ),
        "load_checkpoint must be empty": not str(
            train.get("load_checkpoint", "")
        ).strip(),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid naive multi-Re config: " + "; ".join(failed))


def _warm_start_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    expected_missing = {"log_re_mean", "log_re_std"}
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "Unexpected missing keys while warm-starting naive multi-Re model: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys while warm-starting naive multi-Re model: "
            f"{incompatible.unexpected_keys}"
        )


def _validate_scot_ablation(config):
    recipe = config["train_params"].get("recipe")
    if recipe == "scot_without_tl":
        _validate_scratch_ablation(config)
    elif recipe == "scot_with_tl":
        _validate_transfer_learning_ablation(config)
    elif recipe == "naive_multi_re":
        _validate_naive_multi_re_ablation(config)
    else:
        raise ValueError(
            "train_params.recipe must be scot_without_tl, scot_with_tl, "
            "or naive_multi_re"
        )


def train_scot(config):
    """Train one of the locked three-channel scOT ablations."""
    _validate_scot_ablation(config)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = config["dataset_params"]
    loaders = config["dataloader_params"]
    normalization = {"normalization_params": config["normalization_params"]}
    if dataset.get("dataset_type") == "multi_re":
        train_loader, val_loader, test_loader = get_multi_re_dataloaders(
            dataset,
            normalization_config=normalization,
            batch_size=loaders["train"]["batch_size"],
            num_workers=loaders["train"].get("num_workers", 0),
            seed=dataset.get("seed", 42),
        )
    else:
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

    # Preserve the legacy run order: its shape check consumed one shuffled
    # training iterator before random model initialization.
    probe_inputs, probe_target, probe_metadata = unpack_batch(
        next(iter(train_loader))
    )
    print(f"Input batch shape: {tuple(probe_inputs.shape)}", flush=True)
    print(f"Output batch shape: {tuple(probe_target.shape)}", flush=True)
    del probe_inputs, probe_target, probe_metadata

    model = create_model(config).to(device)
    warm_start_checkpoint = str(
        config["train_params"].get("warm_start_checkpoint", "")
    ).strip()
    if warm_start_checkpoint:
        _warm_start_model(model, warm_start_checkpoint)
    criterion = create_loss(config)
    model_parameters = (
        model.get_optimizer_parameters(config["optimizer_params"])
        if hasattr(model, "get_optimizer_parameters")
        else model.parameters()
    )
    optimizer = create_optimizer(model_parameters, config)
    scheduler = create_scheduler(optimizer, config)
    train = config["train_params"]
    load_checkpoint = str(train.get("load_checkpoint", "")).strip()
    if load_checkpoint:
        raise ValueError(
            "The canonical three-channel scOT ablations start at epoch zero; "
            "load_checkpoint must remain empty."
        )

    best = math.inf
    history = []
    for epoch in range(train["epochs"]):
        train_loss, train_components = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_components = _run_epoch(model, val_loader, criterion, device)
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
            "train_components": train_components,
            "val_components": val_components,
        }
        history.append(row)
        print(row, flush=True)
        if val_loss < best:
            best = val_loss
            _save_checkpoint(
                train["checkpoint_path"],
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                val_rel_l2,
                val_mse,
            )

    checkpoint = torch.load(
        train["checkpoint_path"], map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_components = _run_epoch(model, test_loader, criterion, device)
    test_rel_l2, test_mse = _denormalized_metrics(model, test_loader, device)
    result = {
        "checkpoint_epoch": checkpoint["epoch"],
        "test_loss": test_loss,
        "test_denorm_rel_l2": test_rel_l2,
        "test_denorm_mse": test_mse,
        "test_components": test_components,
        "history": history,
    }
    print(result, flush=True)
    return result
