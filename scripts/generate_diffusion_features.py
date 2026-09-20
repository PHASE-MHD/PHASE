#!/usr/bin/env python3
"""Generate DINO conditioning features from an external tFNO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase.data import get_dataloaders
from phase.models import create_model, map_official_tfno_state_dict
from phase.utils import (
    apply_denormalization,
    get_dataset_normalizer,
    identify_data_channels,
    load_config,
)


def _save_split(path, inputs, targets, sample_ids):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "diff_inputs.npy", inputs)
    np.save(path / "diff_targets.npy", targets)
    np.save(path / "sample_id.npy", np.asarray(sample_ids, dtype=np.int64))


def _generate_split(loader, model, normalizer, output_path, device, sample_ids):
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for inputs, target in loader:
            inputs = inputs.to(device, non_blocking=True).contiguous()
            target = target.to(device, non_blocking=True)
            prediction = model(inputs)
            indices, has_grid, _ = identify_data_channels(inputs, target)
            _, target, prediction = apply_denormalization(
                inputs, target, prediction, normalizer, indices, has_grid
            )
            predictions.append(prediction.cpu().numpy())
            targets.append(target.cpu().numpy())
    condition = np.concatenate(predictions).astype(np.float32, copy=False)
    truth = np.concatenate(targets).astype(np.float32, copy=False)
    if condition.shape[0] != len(sample_ids):
        raise ValueError(
            f"Generated {condition.shape[0]} trajectories but received "
            f"{len(sample_ids)} sample IDs."
        )
    _save_split(output_path, condition, truth, sample_ids)
    return condition.shape


def generate_features(config, checkpoint_path, output_root, batch_size=4, num_workers=1):
    """Generate train/validation/test conditioner and DNS trajectory pairs."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = config["dataset_params"]
    loaders = get_dataloaders(
        dataset["data_path"],
        normalization_config={"normalization_params": config["normalization_params"]},
        batch_size=batch_size,
        num_workers=num_workers,
        train_size=dataset["train_size"],
        val_plus_test_size=dataset["val_plus_test_size"],
        train_sample_fraction=dataset.get("train_sample_fraction", 1.0),
        validation_sample_fraction=dataset.get(
            "validation_sample_fraction", 1.0
        ),
        test_sample_fraction=dataset.get("test_sample_fraction", 1.0),
        seed=dataset.get("seed", 42),
        sub_t=dataset.get("sub_t", 1),
        sub_x=dataset.get("sub_x", 1),
    )
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(map_official_tfno_state_dict(state), strict=True)
    normalizer = get_dataset_normalizer(loaders[0])
    if normalizer is not None:
        normalizer = normalizer.to(device)

    output_root = Path(output_root)
    shapes = {}
    for split, loader in zip(("train", "val", "test"), loaders):
        # Persist features in deterministic source-dataset order.
        loader = DataLoader(
            loader.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
        sample_ids = np.asarray(loader.dataset.indices, dtype=np.int64)
        shapes[split] = _generate_split(
            loader, model, normalizer, output_root / split, device, sample_ids
        )
        print(f"Generated {split}: {shapes[split]}", flush=True)
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    generate_features(
        load_config(args.config),
        args.checkpoint,
        args.output_root,
        args.batch_size,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
