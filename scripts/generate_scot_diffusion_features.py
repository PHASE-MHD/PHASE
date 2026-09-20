#!/usr/bin/env python3
"""Export physical-unit scOT conditions and DNS targets for PHASE diffusion."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase.data import get_dataloaders, get_multi_re_dataloaders
from phase.models import create_model
from phase.utils import get_dataset_normalizer, load_config
from phase.utils.batches import model_forward, move_metadata_to_device, unpack_batch


class DiffusionStore:
    """Incremental directory-backed feature store."""

    def __init__(self, path, count, with_re):
        self.path = Path(path)
        self.count = int(count)
        self.with_re = with_re
        self.offset = 0
        self.arrays = None

    def append(self, predictions, targets, sample_ids, re_values=None):
        if self.arrays is None:
            self.path.mkdir(parents=True, exist_ok=True)
            shape = (self.count,) + tuple(predictions.shape[1:])
            self.arrays = {
                "inputs": np.lib.format.open_memmap(
                    self.path / "diff_inputs.npy", mode="w+", dtype=np.float32, shape=shape
                ),
                "targets": np.lib.format.open_memmap(
                    self.path / "diff_targets.npy", mode="w+", dtype=np.float32, shape=shape
                ),
                "sample_id": np.lib.format.open_memmap(
                    self.path / "sample_id.npy", mode="w+", dtype=np.int64, shape=(self.count,)
                ),
            }
            if self.with_re:
                self.arrays["re"] = np.lib.format.open_memmap(
                    self.path / "re.npy", mode="w+", dtype=np.float32, shape=(self.count,)
                )
        stop = self.offset + len(predictions)
        self.arrays["inputs"][self.offset:stop] = predictions
        self.arrays["targets"][self.offset:stop] = targets
        self.arrays["sample_id"][self.offset:stop] = sample_ids
        if self.with_re:
            self.arrays["re"][self.offset:stop] = re_values
        self.offset = stop

    def close(self):
        if self.offset != self.count:
            raise RuntimeError(f"Wrote {self.offset} items to {self.path}; expected {self.count}")
        for array in self.arrays.values():
            array.flush()
        (self.path / "FORMAT.txt").write_text(
            "PHASE diffusion features v1\n"
            "diff_inputs.npy: scOT prediction [N,C,T,H,W]\n"
            "diff_targets.npy: DNS trajectory [N,C,T,H,W]\n"
            "sample_id.npy: source simulation index [N]\n"
            + ("re.npy: Reynolds number [N]\n" if self.with_re else "")
        )


def _eval_loader(loader, batch_size, workers):
    return DataLoader(
        loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def _denormalize(tensor, normalizer, metadata):
    if normalizer is None:
        return tensor
    try:
        return normalizer.denormalize(tensor, metadata=metadata)
    except TypeError:
        return normalizer.denormalize(tensor)


def _loaders(config, batch_size, workers):
    params = dict(config["dataset_params"])
    normalization = {"normalization_params": config["normalization_params"]}
    if params.get("dataset_type") == "multi_re":
        params["balanced_re_batches"] = False
        loaders = get_multi_re_dataloaders(
            params,
            normalization_config=normalization,
            batch_size=batch_size,
            num_workers=workers,
            seed=params.get("seed", 42),
        )
    else:
        loaders = get_dataloaders(
            params["data_path"],
            normalization_config=normalization,
            batch_size=batch_size,
            num_workers=workers,
            train_size=params["train_size"],
            val_plus_test_size=params["val_plus_test_size"],
            train_sample_fraction=params.get("train_sample_fraction", 1.0),
            validation_sample_fraction=params.get("validation_sample_fraction", 1.0),
            test_sample_fraction=params.get("test_sample_fraction", 1.0),
            seed=params.get("seed", 42),
            sub_t=params.get("sub_t", 1),
            sub_x=params.get("sub_x", 1),
            t_range=tuple(params.get("t_range", (0.0, 1.0))),
            x_range=tuple(params.get("x_range", (0.0, 1.0))),
            y_range=tuple(params.get("y_range", (0.0, 1.0))),
        )
    return tuple(_eval_loader(loader, batch_size, workers) for loader in loaders)


def generate(config, checkpoint_path, output_root, batch_size=4, workers=1, single_re=1000.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = _loaders(config, batch_size, workers)
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    model.eval()
    multi_re = config["dataset_params"].get("dataset_type") == "multi_re"

    for split, loader in zip(("train", "val", "test"), loaders):
        store = DiffusionStore(Path(output_root) / split, len(loader.dataset), multi_re)
        normalizer = get_dataset_normalizer(loader)
        single_ids = np.asarray(getattr(loader.dataset, "indices", []), dtype=np.int64)
        offset = 0
        with torch.no_grad():
            for batch in loader:
                inputs, targets, metadata = unpack_batch(batch)
                size = inputs.shape[0]
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                metadata_device = move_metadata_to_device(metadata, device)
                predictions = model_forward(model, inputs, metadata_device)
                predictions = _denormalize(predictions, normalizer, metadata_device)
                targets = _denormalize(targets, normalizer, metadata_device)
                if predictions.shape[1] != 4 or predictions.shape != targets.shape:
                    raise ValueError(
                        f"Expected matching [N,4,T,H,W], got {predictions.shape} and {targets.shape}"
                    )
                if metadata is None:
                    sample_ids = single_ids[offset:offset + size]
                    re_values = np.full(size, single_re, dtype=np.float32)
                else:
                    sample_ids = metadata["sample_index"].numpy().reshape(-1)
                    re_values = metadata["re"].numpy().reshape(-1)
                store.append(
                    predictions.cpu().float().numpy(),
                    targets.cpu().float().numpy(),
                    sample_ids,
                    re_values,
                )
                offset += size
        store.close()
        print(f"Wrote {split}: {store.path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditioner-config", required=True)
    parser.add_argument("--conditioner-checkpoint", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--single-re", type=float, default=1000.0)
    args = parser.parse_args()
    generate(
        load_config(args.conditioner_config),
        args.conditioner_checkpoint,
        args.output_root,
        args.batch_size,
        args.num_workers,
        args.single_re,
    )


if __name__ == "__main__":
    main()
