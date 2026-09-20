"""Training loop for the locked PHASE scOT recipes."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

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


def _make_tiny_validation_loader(
    val_loader, samples_per_re=5, batch_size=10, seed=42
):
    dataset = val_loader.dataset
    groups = getattr(dataset, "flat_indices_by_re_idx", None)
    if not groups:
        raise ValueError("Tiny per-Re validation requires a MultiReMHDDataset.")
    rng = np.random.default_rng(seed)
    indices = []
    for re_idx in sorted(groups):
        candidates = np.asarray(groups[re_idx], dtype=np.int64)
        if len(candidates) < samples_per_re:
            raise ValueError(
                f"Re index {re_idx} has only {len(candidates)} validation samples."
            )
        indices.extend(
            rng.choice(candidates, size=samples_per_re, replace=False).tolist()
        )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def _should_run_validation(epoch, epochs, interval):
    """Match legacy cadence: first/interval epochs plus the final epoch."""
    return epoch == 0 or epoch % interval == 0 or epoch == epochs - 1


def _tiny_per_re_metrics(model, loader, criterion, device, eps=1e-12):
    model.eval()
    normalizer = copy.deepcopy(getattr(loader.dataset.dataset, "normalizer", None))
    totals = {}
    counts = {}
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
            omega_pred = criterion.curl_2d(prediction[:, 0], prediction[:, 1])
            omega_true = criterion.curl_2d(target[:, 0], target[:, 1])
            current_pred = criterion.curl_2d(prediction[:, 2], prediction[:, 3])
            current_true = criterion.curl_2d(target[:, 2], target[:, 3])
            fields_pred = [prediction[:, i] for i in range(4)] + [
                omega_pred, current_pred
            ]
            fields_true = [target[:, i] for i in range(4)] + [
                omega_true, current_true
            ]
            names = ("ux", "uy", "Bx", "By", "omega", "j")
            sample_metrics = {}
            for name, pred_field, true_field in zip(
                names, fields_pred, fields_true
            ):
                numerator = torch.linalg.vector_norm(
                    (pred_field - true_field).flatten(1), dim=1
                )
                denominator = torch.linalg.vector_norm(
                    true_field.flatten(1), dim=1
                )
                sample_metrics[name] = numerator / (denominator + eps)
            for sample_idx, re_value in enumerate(metadata["re"].tolist()):
                key = str(int(round(re_value)))
                totals.setdefault(key, {name: 0.0 for name in names})
                counts[key] = counts.get(key, 0) + 1
                for name in names:
                    totals[key][name] += float(sample_metrics[name][sample_idx])
    return {
        re_value: {name: value / counts[re_value] for name, value in metrics.items()}
        for re_value, metrics in totals.items()
    }


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
        "magnetic residual learning must be enabled with unit scale": model.get(
            "magnetic_residual", False
        )
        and model.get("magnetic_residual_scale") == 1.0,
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
    normalization = config["normalization_params"]
    dataset = config["dataset_params"]
    loss = config["loss_params"]
    train = config["train_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer.get("param_groups", {})
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
        "magnetic residual learning must be enabled at unit scale": model.get(
            "magnetic_residual", False
        )
        and model.get("magnetic_residual_scale") == 1.0,
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
        "physics normalization must be used": normalization.get("type")
        == "physics",
        "input normalization must be [1, 1, 0.0052]": normalization.get(
            "input_norm"
        )
        == [1.0, 1.0, 0.0052],
        "output normalization must be [1, 1, 0.0052]": normalization.get(
            "output_norm"
        )
        == [1.0, 1.0, 0.0052],
        "dataset_type must be multi_re": dataset.get("dataset_type") == "multi_re",
        "n must be 1000": dataset.get("n") == 1000,
        "the locked ten Re values must be used": dataset.get("re_values")
        == expected_re,
        "Rm values must match Re values": dataset.get("rem_values") == expected_re,
        "balanced Re batches must be enabled": dataset.get(
            "balanced_re_batches", False
        ),
        "res_per_batch must be 10": dataset.get("res_per_batch") == 10,
        "each regime must use an 800/100/100 split": dataset.get(
            "train_size_per_re"
        )
        == 800
        and dataset.get("val_plus_test_size_per_re") == 200,
        "dataset seed must be 42": dataset.get("seed") == 42,
        "sub_t must be 4": dataset.get("sub_t") == 4,
        "sub_x must be 1": dataset.get("sub_x") == 1,
        "the nominal batch_size must be 1 for every split": all(
            value == 1 for value in batch_sizes.values()
        ),
        "physics-informed loss must be used": loss.get("type")
        == "physics-informed",
        "loss fallback nu and eta must both be 1e-3": loss.get("nu") == 1.0e-3
        and loss.get("eta") == 1.0e-3,
        "loss group weights must be [10, 1, 0.001, 0.1]": [
            loss.get(name)
            for name in (
                "data_weight",
                "ic_weight",
                "pde_weight",
                "constraint_weight",
            )
        ]
        == [10.0, 1.0, 0.001, 0.1],
        "field weights must be [1, 1, 5]": [
            loss.get(name) for name in ("u_weight", "v_weight", "A_weight")
        ]
        == [1.0, 1.0, 5.0],
        "PDE weights must be [1, 1, 100]": [
            loss.get(name) for name in ("Du_weight", "Dv_weight", "DA_weight")
        ]
        == [1.0, 1.0, 100.0],
        "divergence weights must be [1, 0]": [
            loss.get(name) for name in ("div_vel_weight", "div_B_weight")
        ]
        == [1.0, 0.0],
        "AdamW with the locked base hyperparameters must be used": optimizer.get(
            "optimizer_type"
        )
        == "adamw"
        and optimizer.get("lr") == 1.0e-5
        and optimizer.get("weight_decay") == 1.0e-2
        and optimizer.get("betas") == [0.9, 0.999],
        "optimizer parameter groups must be enabled": groups.get("enabled", False),
        "pretrained_lr must be 1e-7": groups.get("pretrained_lr") == 1.0e-7,
        "new_lr must be 1e-3": groups.get("new_lr") == 1.0e-3,
        "pretrained_weight_decay must be 1e-2": groups.get(
            "pretrained_weight_decay"
        )
        == 1.0e-2,
        "new_weight_decay must be zero": groups.get("new_weight_decay") == 0.0,
        "the scheduler must be disabled": not optimizer.get(
            "use_scheduler", True
        ),
        "training length and sample fractions must match the declared mode": _valid_dt_training_window(train, dataset),
        "checkpoint selection must use normalized validation loss": train.get(
            "checkpoint_metric"
        )
        == "normalized_validation_loss",
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


def _validate_gated_adapter_multi_re_ablation(config):
    model = config["model_params"]
    normalization = config["normalization_params"]
    dataset = config["dataset_params"]
    loss = config["loss_params"]
    optimizer = config["optimizer_params"]
    train = config["train_params"]
    groups = optimizer.get("param_groups", {})
    conditioning = model.get("re_conditioning", {})
    adapter = conditioning.get("deep_adapter", {})
    expected_re = [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]
    batch_sizes = {
        split: config["dataloader_params"][split]["batch_size"]
        for split in ("train", "validation", "test")
    }
    checks = {
        "model_type must be poseidon-mhd-re-finetune": model.get("model_type")
        == "poseidon-mhd-re-finetune",
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
        "the locked 128-grid, patch-4, three-channel model must be used": [
            model.get(name) for name in ("image_size", "patch_size", "out_channels")
        ]
        == [128, 4, 3],
        "the POSEIDON velocity channel maps must both be [1, 2]": model.get(
            "poseidon_input_channel_map"
        )
        == [1, 2]
        and model.get("poseidon_output_channel_map") == [1, 2],
        "the magnetic channel index must be 4": model.get("magnetic_channel_index")
        == 4,
        "magnetic boundary initialization must be mean-velocity input and zero output": model.get(
            "magnetic_input_init"
        )
        == "mean_velocity"
        and model.get("magnetic_output_init") == "zero",
        "Helmholtz projection must not be enabled for the vector-potential ablation": not model.get(
            "helmholtz_projection", False
        ),
        "velocity residual learning must be disabled": not model.get(
            "velocity_residual", False
        ),
        "magnetic residual learning must be enabled": model.get(
            "magnetic_residual", False
        ),
        "gated conditioning must be enabled": conditioning.get("enabled", False),
        "conditioning type must be deep_adapter_output_film": conditioning.get(
            "type"
        )
        == "deep_adapter_output_film",
        "FiLM dimensions must be hidden_dim=128 and num_layers=2": conditioning.get(
            "hidden_dim"
        )
        == 128
        and conditioning.get("num_layers") == 2,
        "Re/Rm conditioning must include Rm": conditioning.get(
            "include_rem", False
        ),
        "log_re_mean must be 2.9756": conditioning.get("log_re_mean") == 2.9756,
        "log_re_std must be 0.5417": conditioning.get("log_re_std") == 0.5417,
        "conditioning residual scale must be 1": conditioning.get(
            "residual_scale"
        )
        == 1.0,
        "reference_re must be 1000": conditioning.get("reference_re") == 1000,
        "deep adapters must target all blocks": adapter.get("target") == "all",
        "adapter bottleneck must be 64": adapter.get("bottleneck_dim") == 64,
        "adapter MLP must use hidden_dim=128 and num_layers=2": adapter.get(
            "hidden_dim"
        )
        == 128
        and adapter.get("num_layers") == 2,
        "adapter scale must be 1": adapter.get("adapter_scale") == 1.0,
        "adapter gate type must be channel": adapter.get("gate_type") == "channel",
        "physics normalization must be used": normalization.get("type")
        == "physics",
        "input normalization must be [1, 1, 0.0052]": normalization.get(
            "input_norm"
        )
        == [1.0, 1.0, 0.0052],
        "output normalization must be [1, 1, 0.0052]": normalization.get(
            "output_norm"
        )
        == [1.0, 1.0, 0.0052],
        "dataset_type must be multi_re": dataset.get("dataset_type") == "multi_re",
        "data_file must be mhd_data_3channel.npy": dataset.get("data_file")
        == "mhd_data_3channel.npy",
        "n must be 1000": dataset.get("n") == 1000,
        "the locked ten Re values must be used": dataset.get("re_values")
        == expected_re,
        "Rm values must match Re values": dataset.get("rem_values") == expected_re,
        "balanced Re batches must be enabled": dataset.get(
            "balanced_re_batches", False
        ),
        "res_per_batch must be 10": dataset.get("res_per_batch") == 10,
        "each regime must use an 800/100/100 split": dataset.get(
            "train_size_per_re"
        )
        == 800
        and dataset.get("val_plus_test_size_per_re") == 200,
        "dataset seed must be 42": dataset.get("seed") == 42,
        "sub_t must be 4": dataset.get("sub_t") == 4,
        "sub_x must be 1": dataset.get("sub_x") == 1,
        "the unit space-time domain must be used": dataset.get("t_range")
        == [0.0, 1.0]
        and dataset.get("x_range") == [0.0, 1.0]
        and dataset.get("y_range") == [0.0, 1.0],
        "the nominal batch_size must be 1 for every split": all(
            value == 1 for value in batch_sizes.values()
        ),
        "physics-informed loss must be used": loss.get("type")
        == "physics-informed",
        "all four configured loss terms must be enabled": all(
            loss.get(name, False)
            for name in (
                "use_data_loss",
                "use_ic_loss",
                "use_pde_loss",
                "use_constraint_loss",
            )
        ),
        "loss fallback nu and eta must both be 1e-3": loss.get("nu") == 1.0e-3
        and loss.get("eta") == 1.0e-3,
        "loss group weights must be [10, 1, 0.001, 0.1]": [
            loss.get(name)
            for name in (
                "data_weight",
                "ic_weight",
                "pde_weight",
                "constraint_weight",
            )
        ]
        == [10.0, 1.0, 0.001, 0.1],
        "field weights must be [1, 1, 5]": [
            loss.get(name) for name in ("u_weight", "v_weight", "A_weight")
        ]
        == [1.0, 1.0, 5.0],
        "PDE weights must be [1, 1, 100]": [
            loss.get(name) for name in ("Du_weight", "Dv_weight", "DA_weight")
        ]
        == [1.0, 1.0, 100.0],
        "divergence weights must be [1, 0]": [
            loss.get(name) for name in ("div_vel_weight", "div_B_weight")
        ]
        == [1.0, 0.0],
        "the loss domain and density must be unity": loss.get("rho0") == 1.0
        and loss.get("Lx") == 1.0
        and loss.get("Ly") == 1.0
        and loss.get("tend") == 1.0,
        "weighted means must be disabled": not loss.get("use_weighted_mean", True),
        "AdamW with the locked base hyperparameters must be used": optimizer.get(
            "optimizer_type"
        )
        == "adamw"
        and optimizer.get("lr") == 1.0e-5
        and optimizer.get("weight_decay") == 1.0e-2
        and optimizer.get("betas") == [0.9, 0.999],
        "optimizer parameter groups must be enabled": groups.get("enabled", False),
        "pretrained_lr must be 1e-7": groups.get("pretrained_lr") == 1.0e-7,
        "new_lr must be 1e-3": groups.get("new_lr") == 1.0e-3,
        "boundary parameters must remain in the pretrained group": groups.get(
            "boundary_group"
        )
        == "pretrained",
        "pretrained_weight_decay must be 1e-2": groups.get(
            "pretrained_weight_decay"
        )
        == 1.0e-2,
        "new_weight_decay must be zero": groups.get("new_weight_decay") == 0.0,
        "pretrained parameters must remain trainable": not groups.get(
            "freeze_pretrained", True
        ),
        "the scheduler must be disabled": not optimizer.get(
            "use_scheduler", True
        ),
        "training length and sample fractions must match the declared mode": _valid_dt_training_window(train, dataset),
        "the run must remain a fine-tune": train.get("is_finetune", False),
        "checkpoint selection must use normalized validation loss": train.get(
            "checkpoint_metric"
        )
        == "normalized_validation_loss",
        "warm_start_checkpoint must be set": bool(
            str(train.get("warm_start_checkpoint", "")).strip()
        ),
        "load_checkpoint must be empty": not str(
            train.get("load_checkpoint", "")
        ).strip(),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "Invalid gated-adapter multi-Re config: " + "; ".join(failed)
        )


def _warm_start_model(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=False)
    if not hasattr(model, "expected_warm_start_missing_keys"):
        raise RuntimeError(
            "Warm-started model must declare expected_warm_start_missing_keys()."
        )
    expected_missing = set(model.expected_warm_start_missing_keys())
    if set(incompatible.missing_keys) != expected_missing:
        raise RuntimeError(
            "Unexpected missing keys while warm-starting multi-Re model: "
            f"{incompatible.missing_keys}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint keys while warm-starting multi-Re model: "
            f"{incompatible.unexpected_keys}"
        )


def _validate_four_channel_common(
    config,
    expected_norm=None,
    expected_tend=1.0,
):
    model = config["model_params"]
    normalization = config["normalization_params"]
    loss = config["loss_params"]
    if expected_norm is None:
        expected_norm = [1.0, 1.0, 4.27121774e-03, 4.27121774e-03]
    checks = {
        "the canonical pretrained POSEIDON backbone must be used": model.get(
            "poseidon_model"
        ) == "camlab-ethz/Poseidon-T"
        and model.get("load_pretrained_poseidon", False),
        "POSEIDON velocity channels must map to [1, 2]": model.get(
            "poseidon_input_channel_map"
        ) == [1, 2]
        and model.get("poseidon_output_channel_map") == [1, 2],
        "the expanded POSEIDON fallback must have six channels": model.get(
            "fallback_poseidon_num_channels"
        ) == 6
        and model.get("fallback_poseidon_num_out_channels") == 6,
        "the model must predict four physical fields [u, v, Bx, By]": model.get(
            "out_channels"
        )
        == 4,
        "magnetic channels must map to POSEIDON channels [4, 5]": model.get(
            "magnetic_channel_indices"
        ) == [4, 5],
        "POSEIDON-native velocity normalization must be enabled": model.get(
            "use_poseidon_fluid_normalization", False
        ),
        "magnetic input/output initialization must use mean velocity": model.get(
            "magnetic_input_init"
        ) == "mean_velocity"
        and model.get("magnetic_output_init") == "mean_velocity",
        "velocity residual learning must be disabled": not model.get(
            "velocity_residual", False
        ),
        "magnetic residual learning must be enabled at unit scale": model.get(
            "magnetic_residual", False
        )
        and model.get("magnetic_residual_scale") == 1.0,
        "Helmholtz projection must cover velocity and magnetic pairs": model.get(
            "helmholtz_projection", False
        )
        and model.get("project_velocity", False)
        and model.get("project_magnetic", False),
        "Helmholtz domain lengths must be unity": model.get(
            "helmholtz_domain_size_x"
        ) == 1.0
        and model.get("helmholtz_domain_size_y") == 1.0,
        "paired physics normalization must use the locked global scale": normalization.get(
            "type"
        ) == "physics"
        and normalization.get("input_norm") == expected_norm
        and normalization.get("output_norm") == expected_norm,
        "velocity and magnetic components must remain paired": normalization.get(
            "paired_input_groups"
        ) == [[0, 1], [2, 3]]
        and normalization.get("paired_output_groups") == [[0, 1], [2, 3]]
        and normalization.get("paired_mode") == "rms",
        "the direct-B physics objective must be used": loss.get("type")
        == "physics-informed-bfield",
        "fallback nu and eta must both be 1e-3": loss.get("nu") == 1.0e-3
        and loss.get("eta") == 1.0e-3,
        "loss group weights must be [10, 1, 0.001, 0]": [
            loss.get(name)
            for name in ("data_weight", "ic_weight", "pde_weight", "constraint_weight")
        ] == [10.0, 1.0, 0.001, 0.0],
        "vorticity/current weights must be [2, 5]": [
            loss.get("vorticity_weight"), loss.get("current_weight")
        ] == [2.0, 5.0],
        "field weights must be [1, 1, 5, 5]": [
            loss.get(name) for name in ("u_weight", "v_weight", "Bx_weight", "By_weight")
        ] == [1.0, 1.0, 5.0, 5.0],
        "PDE weights must be [1, 1, 100, 100]": [
            loss.get(name) for name in ("Du_weight", "Dv_weight", "DBx_weight", "DBy_weight")
        ] == [1.0, 1.0, 100.0, 100.0],
        "soft divergence penalties must be disabled": [
            loss.get("constraint_weight"), loss.get("div_vel_weight"), loss.get("div_B_weight")
        ] == [0.0, 0.0, 0.0],
        "all configured data, IC, PDE, and constraint branches must remain enabled": all(
            loss.get(name, False)
            for name in ("use_data_loss", "use_ic_loss", "use_pde_loss", "use_constraint_loss")
        ),
        "the unit space-time domain must be used": loss.get("Lx") == 1.0
        and loss.get("Ly") == 1.0
        and loss.get("tend") == expected_tend,
        "weighted means must be disabled": not loss.get("use_weighted_mean", True),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid four-channel PHASE config: " + "; ".join(failed))


def _valid_dt_training_window(train, dataset):
    """Keep reduced acceptance runs explicit and production recipes locked."""
    fractions = [
        dataset.get("train_sample_fraction", 1.0),
        dataset.get("validation_sample_fraction", 1.0),
        dataset.get("test_sample_fraction", 1.0),
    ]
    if train.get("acceptance_run") is True:
        return train.get("epochs") == 10 and fractions == [0.2, 0.1, 0.1]
    return train.get("epochs") == 100 and fractions == [1.0, 1.0, 1.0]


def _validate_four_channel_single_re(config):
    _validate_four_channel_common(config)
    dataset = config["dataset_params"]
    loaders = config["dataloader_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer.get("param_groups", {})
    train = config["train_params"]
    checks = {
        "the single-Re POSEIDON model type must be selected": config[
            "model_params"
        ].get("model_type") == "poseidon-mhd-finetune",
        "single-Re data must use the four-channel Re=1000 array": dataset.get(
            "dataset_type"
        ) == "single_re"
        and dataset.get("data_path", "").endswith(
            "/mhd_Re1000_N1000/mhd_data_4channel.npy"
        ),
        "the split must be 800/100/100 with seed 42": dataset.get("train_size") == 800
        and dataset.get("val_plus_test_size") == 200
        and dataset.get("seed") == 42,
        "single-Re temporal/spatial subsampling must be [4, 1]": [
            dataset.get("sub_t"), dataset.get("sub_x")
        ] == [4, 1],
        "single-Re batch size must be 16 for every split": all(
            loaders[split].get("batch_size") == 16
            for split in ("train", "validation", "test")
        ),
        "single-Re AdamW settings must be canonical": optimizer.get(
            "optimizer_type"
        ) == "adamw"
        and optimizer.get("lr") == 1.0e-5
        and optimizer.get("weight_decay") == 1.0e-2
        and optimizer.get("betas") == [0.9, 0.999]
        and groups.get("enabled", False),
        "single-Re copied/new learning rates must be [5e-6, 5e-4]": [
            groups.get("pretrained_lr"), groups.get("new_lr")
        ] == [5.0e-6, 5.0e-4],
        "magnetic output effective LR must be 2e-3": groups.get(
            "magnetic_output_lr"
        ) == 2.0e-3,
        "single-Re weight decays must be [1e-2, 0]": [
            groups.get("pretrained_weight_decay"), groups.get("new_weight_decay")
        ] == [1.0e-2, 0.0],
        "the scheduler must be disabled": not optimizer.get("use_scheduler", True),
        "single-Re training length must match its declared mode": _valid_dt_training_window(
            train, dataset
        ) and train.get("is_finetune", False)
        and not str(train.get("warm_start_checkpoint", "")).strip()
        and not str(train.get("load_checkpoint", "")).strip(),
        "checkpoint selection must use normalized validation loss": train.get(
            "checkpoint_metric"
        ) == "normalized_validation_loss",
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid four-channel single-Re config: " + "; ".join(failed))


def _validate_four_channel_multi_re(config):
    _validate_four_channel_common(config)
    model = config["model_params"]
    conditioning = model.get("re_conditioning", {})
    adapter = conditioning.get("deep_adapter", {})
    dataset = config["dataset_params"]
    loaders = config["dataloader_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer.get("param_groups", {})
    train = config["train_params"]
    expected_re = [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]
    checks = {
        "the gated multi-Re model type must be selected": model.get("model_type")
        == "poseidon-mhd-re-finetune",
        "deep-adapter output FiLM conditioning must be enabled": conditioning.get(
            "enabled", False
        )
        and conditioning.get("type") == "deep_adapter_output_film"
        and conditioning.get("hidden_dim") == 128
        and conditioning.get("num_layers") == 2
        and conditioning.get("residual_scale") == 1.0,
        "conditioning must use standardized log10(Re,Rm)": conditioning.get(
            "include_rem", False
        )
        and conditioning.get("log_re_mean") == 2.9756
        and conditioning.get("log_re_std") == 0.5417
        and conditioning.get("reference_re") == 1000.0,
        "all-block channel-gated adapters must use the canonical dimensions": adapter.get(
            "target"
        ) == "all"
        and adapter.get("bottleneck_dim") == 64
        and adapter.get("hidden_dim") == 128
        and adapter.get("num_layers") == 2
        and adapter.get("adapter_scale") == 1.0
        and adapter.get("gate_type") == "channel",
        "the ten locked Re=Rm regimes must be used": dataset.get("re_values")
        == expected_re
        and dataset.get("rem_values") == expected_re,
        "multi-Re data must use four-channel arrays": dataset.get("dataset_type")
        == "multi_re"
        and dataset.get("data_file") == "mhd_data_4channel.npy",
        "every regime must use an 800/100/100 split": dataset.get(
            "train_size_per_re"
        ) == 800
        and dataset.get("val_plus_test_size_per_re") == 200
        and dataset.get("seed") == 42,
        "balanced ten-regime batches must be enabled": dataset.get(
            "balanced_re_batches", False
        )
        and dataset.get("res_per_batch") == 10,
        "multi-Re temporal/spatial subsampling must be [4, 1]": [
            dataset.get("sub_t"), dataset.get("sub_x")
        ] == [4, 1],
        "nominal multi-Re batch size must be 1": all(
            loaders[split].get("batch_size") == 1
            for split in ("train", "validation", "test")
        ),
        "multi-Re AdamW settings must be canonical": optimizer.get(
            "optimizer_type"
        ) == "adamw"
        and optimizer.get("lr") == 1.0e-5
        and optimizer.get("weight_decay") == 1.0e-2
        and optimizer.get("betas") == [0.9, 0.999]
        and groups.get("enabled", False),
        "multi-Re copied/adapter learning rates must be [1e-7, 1e-3]": [
            groups.get("pretrained_lr"), groups.get("new_lr")
        ] == [1.0e-7, 1.0e-3],
        "the recorded magnetic-output compatibility value must remain 2e-3": groups.get(
            "magnetic_output_lr"
        ) == 2.0e-3,
        "expanded boundary tensors must remain trainable in the pretrained group": groups.get(
            "boundary_group"
        ) == "pretrained"
        and not groups.get("freeze_pretrained", True),
        "multi-Re weight decays must be [1e-2, 0]": [
            groups.get("pretrained_weight_decay"), groups.get("new_weight_decay")
        ] == [1.0e-2, 0.0],
        "the scheduler must be disabled": not optimizer.get("use_scheduler", True),
        "multi-Re training must warm-start weights but begin at epoch zero": (
            _valid_dt_training_window(train, dataset)
        )
        and train.get("is_finetune", False)
        and bool(str(train.get("warm_start_checkpoint", "")).strip())
        and not str(train.get("load_checkpoint", "")).strip(),
        "checkpoint selection must use normalized validation loss": train.get(
            "checkpoint_metric"
        ) == "normalized_validation_loss",
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid four-channel multi-Re config: " + "; ".join(failed))


def _validate_kh_common(config, expected_norm):
    _validate_four_channel_common(
        config, expected_norm=expected_norm, expected_tend=5.0
    )
    dataset = config["dataset_params"]
    loss = config["loss_params"]
    checks = {
        "KH data must retain all 51 frames after sub_t=5": dataset.get("sub_t") == 5
        and dataset.get("sub_x") == 1
        and dataset.get("t_range") == [0.0, 5.0],
        "all KH primary and derived losses must be time-relative": all(
            loss.get(name) == "time_relative"
            for name in (
                "u_time_loss_mode",
                "v_time_loss_mode",
                "magnetic_time_loss_mode",
                "derived_time_loss_mode",
            )
        ),
        "all KH time-relative epsilons must be 1e-6": all(
            loss.get(name) == 1.0e-6
            for name in (
                "u_time_loss_eps",
                "v_time_loss_eps",
                "magnetic_time_loss_eps",
                "derived_time_loss_eps",
            )
        ),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid KH scOT config: " + "; ".join(failed))


def _validate_kh_single_re(config):
    expected_norm = [1.0, 1.0, 6.69424514e-02, 6.69424514e-02]
    _validate_kh_common(config, expected_norm)
    model = config["model_params"]
    dataset = config["dataset_params"]
    loaders = config["dataloader_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer.get("param_groups", {})
    train = config["train_params"]
    checks = {
        "the single-Re KH model type must be selected": model.get("model_type")
        == "poseidon-mhd-finetune",
        "single-Re KH data must be the Re=1000 four-channel array": dataset.get(
            "dataset_type"
        ) == "single_re"
        and dataset.get("data_path", "").endswith(
            "/mhd_Re1000_N1000/mhd_data_4channel.npy"
        ),
        "the KH split must be 800/100/100 with seed 42": dataset.get("train_size")
        == 800
        and dataset.get("val_plus_test_size") == 200
        and dataset.get("seed") == 42,
        "single-Re KH batch size must be one": all(
            loaders[split].get("batch_size") == 1
            for split in ("train", "validation", "test")
        ),
        "single-Re KH optimizer groups must match the selected run": optimizer.get(
            "optimizer_type"
        ) == "adamw"
        and optimizer.get("lr") == 1.0e-5
        and optimizer.get("weight_decay") == 1.0e-2
        and optimizer.get("betas") == [0.9, 0.999]
        and groups.get("enabled", False)
        and groups.get("pretrained_lr") == 5.0e-6
        and groups.get("new_lr") == 5.0e-4
        and groups.get("magnetic_output_lr") == 2.0e-3
        and groups.get("pretrained_weight_decay") == 1.0e-2
        and groups.get("new_weight_decay") == 0.0
        and not optimizer.get("use_scheduler", True),
        "single-Re KH training window must match the declared mode": (
            _valid_dt_training_window(train, dataset)
        )
        and train.get("is_finetune", False)
        and not str(train.get("warm_start_checkpoint", "")).strip()
        and not str(train.get("load_checkpoint", "")).strip(),
        "single-Re KH checkpointing must use normalized validation loss": train.get(
            "checkpoint_metric"
        ) == "normalized_validation_loss",
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid single-Re KH scOT config: " + "; ".join(failed))


def _validate_kh_multi_re(config):
    expected_norm = [1.0, 1.0, 8.06614549e-02, 8.06614549e-02]
    _validate_kh_common(config, expected_norm)
    model = config["model_params"]
    conditioning = model.get("re_conditioning", {})
    adapter = conditioning.get("deep_adapter", {})
    dataset = config["dataset_params"]
    loaders = config["dataloader_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer.get("param_groups", {})
    train = config["train_params"]
    expected_re = [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]
    checks = {
        "the multi-Re KH gated model type must be selected": model.get("model_type")
        == "poseidon-mhd-re-finetune",
        "KH conditioning must use the all-block channel-gated adapter": conditioning.get(
            "enabled", False
        )
        and conditioning.get("type") == "deep_adapter_output_film"
        and conditioning.get("hidden_dim") == 128
        and conditioning.get("num_layers") == 2
        and conditioning.get("residual_scale") == 1.0
        and conditioning.get("include_rem", False)
        and conditioning.get("log_re_mean") == 2.9756
        and conditioning.get("log_re_std") == 0.5417
        and conditioning.get("reference_re") == 1000.0
        and adapter.get("target") == "all"
        and adapter.get("bottleneck_dim") == 64
        and adapter.get("hidden_dim") == 128
        and adapter.get("num_layers") == 2
        and adapter.get("adapter_scale") == 1.0
        and adapter.get("gate_type") == "channel",
        "the ten locked KH Re=Rm regimes must be used": dataset.get("re_values")
        == expected_re
        and dataset.get("rem_values") == expected_re,
        "multi-Re KH data must use four-channel arrays": dataset.get("dataset_type")
        == "multi_re"
        and dataset.get("data_file") == "mhd_data_4channel.npy",
        "every KH regime must use an 800/100/100 split": dataset.get(
            "train_size_per_re"
        ) == 800
        and dataset.get("val_plus_test_size_per_re") == 200
        and dataset.get("seed") == 42,
        "KH batches must balance all ten regimes": dataset.get(
            "balanced_re_batches", False
        )
        and dataset.get("res_per_batch") == 10,
        "nominal multi-Re KH batch size must be one": all(
            loaders[split].get("batch_size") == 1
            for split in ("train", "validation", "test")
        ),
        "multi-Re KH loaders must use four workers": all(
            loaders[split].get("num_workers") == 4
            for split in ("train", "validation", "test")
        ),
        "multi-Re KH optimizer groups must match the selected run": optimizer.get(
            "optimizer_type"
        ) == "adamw"
        and optimizer.get("lr") == 1.0e-5
        and optimizer.get("weight_decay") == 1.0e-2
        and optimizer.get("betas") == [0.9, 0.999]
        and groups.get("enabled", False)
        and groups.get("pretrained_lr") == 1.0e-7
        and groups.get("new_lr") == 1.0e-3
        and groups.get("magnetic_output_lr") == 2.0e-3
        and groups.get("boundary_group") == "pretrained"
        and not groups.get("freeze_pretrained", True)
        and groups.get("pretrained_weight_decay") == 1.0e-2
        and groups.get("new_weight_decay") == 0.0
        and not optimizer.get("use_scheduler", True),
        "multi-Re KH must warm-start weights and begin at epoch zero": (
            _valid_dt_training_window(train, dataset)
        )
        and not train.get("is_finetune", True)
        and bool(str(train.get("warm_start_checkpoint", "")).strip())
        and not str(train.get("load_checkpoint", "")).strip(),
        "multi-Re KH checkpointing must use full denormalized relative L2": train.get(
            "checkpoint_metric"
        ) == "denorm_rel_l2"
        and train.get("validation_interval") == 5,
        "tiny KH validation must match the declared mode": (
            train.get("acceptance_run") is True
            and all(
                key not in train
                for key in (
                    "tiny_validation_interval",
                    "tiny_validation_samples_per_re",
                    "tiny_validation_batch_size",
                    "tiny_validation_seed",
                )
            )
        )
        or (
            train.get("acceptance_run") is not True
            and train.get("tiny_validation_interval") == 1
            and train.get("tiny_validation_samples_per_re") == 5
            and train.get("tiny_validation_batch_size") == 10
            and train.get("tiny_validation_seed") == 42
        ),
    }
    failed = [message for message, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Invalid multi-Re KH scOT config: " + "; ".join(failed))


def _validate_scot_ablation(config):
    recipe = config["train_params"].get("recipe")
    if recipe == "scot_without_tl":
        _validate_scratch_ablation(config)
    elif recipe == "scot_with_tl":
        _validate_transfer_learning_ablation(config)
    elif recipe == "naive_multi_re":
        _validate_naive_multi_re_ablation(config)
    elif recipe == "gated_adapter_multi_re":
        _validate_gated_adapter_multi_re_ablation(config)
    elif recipe == "four_channel_single_re":
        _validate_four_channel_single_re(config)
    elif recipe == "four_channel_multi_re":
        _validate_four_channel_multi_re(config)
    elif recipe == "kh_four_channel_single_re":
        _validate_kh_single_re(config)
    elif recipe == "kh_four_channel_multi_re":
        _validate_kh_multi_re(config)
    else:
        raise ValueError(
            "train_params.recipe must be scot_without_tl, scot_with_tl, "
            "naive_multi_re, gated_adapter_multi_re, four_channel_single_re, "
            "four_channel_multi_re, kh_four_channel_single_re, or kh_four_channel_multi_re"
        )


def train_scot(config):
    """Train one of the locked PHASE scOT recipes."""
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
            train_sample_fraction=dataset.get("train_sample_fraction", 1.0),
            validation_sample_fraction=dataset.get("validation_sample_fraction", 1.0),
            test_sample_fraction=dataset.get("test_sample_fraction", 1.0),
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
            "Canonical scOT recipes start at epoch zero; "
            "load_checkpoint must remain empty."
        )

    validation_interval = int(train.get("validation_interval", 1))
    tiny_interval = int(train.get("tiny_validation_interval", 0))
    checkpoint_metric = train.get(
        "checkpoint_metric", "normalized_validation_loss"
    )
    if checkpoint_metric not in {"normalized_validation_loss", "denorm_rel_l2"}:
        raise ValueError(
            "checkpoint_metric must be normalized_validation_loss or denorm_rel_l2."
        )

    best = math.inf
    history = []
    for epoch in range(train["epochs"]):
        train_loss, train_components = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        run_full_validation = _should_run_validation(
            epoch, train["epochs"], validation_interval
        )
        val_loss = val_rel_l2 = val_mse = None
        val_components = {}
        if run_full_validation:
            val_loss, val_components = _run_epoch(
                model, val_loader, criterion, device
            )
            val_rel_l2, val_mse = _denormalized_metrics(
                model, val_loader, device
            )

        tiny_metrics = None
        run_tiny_validation = tiny_interval > 0 and _should_run_validation(
            epoch, train["epochs"], tiny_interval
        )
        if run_tiny_validation:
            tiny_loader = _make_tiny_validation_loader(
                val_loader,
                samples_per_re=int(
                    train.get("tiny_validation_samples_per_re", 5)
                ),
                batch_size=int(train.get("tiny_validation_batch_size", 10)),
                seed=int(train.get("tiny_validation_seed", 42)) + epoch,
            )
            tiny_metrics = _tiny_per_re_metrics(
                model, tiny_loader, criterion, device
            )

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if run_full_validation:
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
            "tiny_val_per_re": tiny_metrics,
        }
        history.append(row)
        print(row, flush=True)

        if run_full_validation:
            score = (
                val_loss
                if checkpoint_metric == "normalized_validation_loss"
                else val_rel_l2
            )
            if score < best:
                best = score
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
        train["checkpoint_path"], map_location=device, weights_only=True
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
