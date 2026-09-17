"""Multi-Re MHD datasets for neural-operator training."""

import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..normalizations import create_normalization


def _load_normalization_config(normalization_config):
    if not normalization_config or "normalization_params" not in normalization_config:
        return normalization_config

    norm_params = normalization_config["normalization_params"]
    if "stats_file" not in norm_params or not norm_params["stats_file"].endswith(".npz"):
        return normalization_config

    stats_file = norm_params["stats_file"]
    if not os.path.exists(stats_file):
        raise FileNotFoundError(f"Statistics file not found: {stats_file}")

    print(f"Loading normalization statistics from {stats_file}")
    stats = np.load(stats_file)
    processed = {"normalization_params": norm_params.copy()}
    for key in stats.files:
        processed["normalization_params"][key] = (
            stats[key].tolist() if hasattr(stats[key], "tolist") else stats[key]
        )
        print(f"{key}: {stats[key]}")
    return processed


class MultiReMHDDataset(Dataset):
    """Dataset that mixes multiple Reynolds-number MHD datasets.

    Each item returns ``(inputs, outputs, metadata)`` where ``metadata`` contains
    raw ``re``, ``rem``, and the corresponding ``nu``/``eta`` coefficients.
    Splits are created independently inside each Re dataset.
    """

    def __init__(
        self,
        datasets: Sequence[torch.Tensor],
        re_values: Sequence[float],
        rem_values: Optional[Sequence[float]] = None,
        normalization_config=None,
        split: str = "train",
        train_size_per_re: Optional[int] = None,
        val_plus_test_size_per_re: Optional[int] = None,
        seed: int = 42,
        sub_t: int = 1,
        sub_x: int = 1,
        t_range=(0, 1.0),
        x_range=(0, 1.0),
        y_range=(0, 1.0),
    ):
        if len(datasets) != len(re_values):
            raise ValueError("datasets and re_values must have the same length.")
        if rem_values is None:
            rem_values = re_values
        if len(rem_values) != len(re_values):
            raise ValueError("rem_values and re_values must have the same length.")
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be 'train', 'val', or 'test'.")

        self.datasets = list(datasets)
        self.re_values = [float(value) for value in re_values]
        self.rem_values = [float(value) for value in rem_values]
        self.split = split
        self.sub_t = sub_t
        self.sub_x = sub_x

        first = self.datasets[0]
        self.channel_last = first.ndim == 5 and first.shape[-1] in (3, 4)
        if self.channel_last:
            self.channels = int(first.shape[-1])
            self.timesteps = int(first.shape[1])
            self.base_nx = int(first.shape[2])
            self.base_ny = int(first.shape[3])
            expected_shape = first.shape[1:]
        else:
            self.channels = int(first.shape[1])
            self.timesteps = int(first.shape[2])
            self.base_nx = int(first.shape[3])
            self.base_ny = int(first.shape[4])
            expected_shape = first.shape[1:]

        self.normalizer = None
        if normalization_config:
            self.normalizer = create_normalization(normalization_config)

        self.samples: List[tuple[int, int]] = []
        self.flat_indices_by_re_idx: Dict[int, List[int]] = {
            re_idx: [] for re_idx in range(len(self.datasets))
        }

        for re_idx, data in enumerate(self.datasets):
            if data.shape[1:] != expected_shape:
                raise ValueError(
                    "All Re datasets must have matching channel/time/spatial shapes. "
                    f"Dataset 0 has {tuple(first.shape)}, dataset {re_idx} has "
                    f"{tuple(data.shape)}."
                )

            num_samples = int(data.shape[0])
            train_size = (
                train_size_per_re if train_size_per_re is not None else num_samples
            )
            val_plus_test = (
                val_plus_test_size_per_re
                if val_plus_test_size_per_re is not None
                else max(num_samples - train_size, 0)
            )
            if train_size + val_plus_test > num_samples:
                raise ValueError(
                    f"Requested train_size_per_re + val_plus_test_size_per_re = "
                    f"{train_size + val_plus_test}, but Re={self.re_values[re_idx]} "
                    f"only has {num_samples} samples."
                )

            rng = np.random.default_rng(seed + re_idx)
            shuffled = rng.permutation(num_samples)
            val_size = val_plus_test // 2

            if split == "train":
                selected = shuffled[:train_size]
            elif split == "val":
                selected = shuffled[train_size : train_size + val_size]
            else:
                selected = shuffled[train_size + val_size : train_size + val_plus_test]

            for sample_idx in selected:
                flat_idx = len(self.samples)
                self.samples.append((re_idx, int(sample_idx)))
                self.flat_indices_by_re_idx[re_idx].append(flat_idx)

        print(
            f"Initialized {split} multi-Re dataset with {len(self.samples)} samples "
            f"across Re={self.re_values}"
        )

        self.t = np.linspace(t_range[0], t_range[1], self.timesteps)[::sub_t]
        self.x = np.linspace(x_range[0], x_range[1], self.base_nx)[::sub_x]
        self.y = np.linspace(y_range[0], y_range[1], self.base_ny)[::sub_x]

        self.nt = len(self.t)
        self.nx = len(self.x)
        self.ny = len(self.y)

        t_tensor = torch.tensor(self.t, dtype=torch.float32)
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.float32)
        self.grid_t = t_tensor.reshape(self.nt, 1, 1, 1).repeat(1, self.nx, self.ny, 1)
        self.grid_x = x_tensor.reshape(1, self.nx, 1, 1).repeat(self.nt, 1, self.ny, 1)
        self.grid_y = y_tensor.reshape(1, 1, self.ny, 1).repeat(self.nt, self.nx, 1, 1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        re_idx, sample_idx = self.samples[idx]
        raw = self.datasets[re_idx]
        if self.channel_last:
            data_np = raw[
                sample_idx, :: self.sub_t, :: self.sub_x, :: self.sub_x, :
            ]
            data = torch.as_tensor(
                np.array(data_np, dtype=np.float32, copy=True)
            ).permute(3, 0, 1, 2)
        else:
            data_np = raw[
                sample_idx, :, :: self.sub_t, :: self.sub_x, :: self.sub_x
            ]
            data = torch.as_tensor(np.array(data_np, dtype=np.float32, copy=True))

        re = self.re_values[re_idx]
        rem = self.rem_values[re_idx]
        metadata = {
            "re": torch.tensor(re, dtype=torch.float32),
            "rem": torch.tensor(rem, dtype=torch.float32),
            "nu": torch.tensor(1.0 / re, dtype=torch.float32),
            "eta": torch.tensor(1.0 / rem, dtype=torch.float32),
            "re_index": torch.tensor(re_idx, dtype=torch.long),
            "sample_index": torch.tensor(sample_idx, dtype=torch.long),
        }

        if self.normalizer:
            data = self.normalizer.normalize(data.unsqueeze(0), metadata=metadata).squeeze(0)

        data = data.float()
        data0 = data[:, 0:1].repeat(1, self.nt, 1, 1).permute(1, 2, 3, 0)
        inputs = torch.cat(
            [self.grid_t, self.grid_x, self.grid_y, data0], dim=-1
        ).permute(3, 0, 1, 2)
        outputs = data

        return inputs, outputs, metadata


class BalancedReBatchSampler(Sampler[List[int]]):
    """Batch sampler that keeps Re values represented throughout an epoch."""

    def __init__(
        self,
        dataset: MultiReMHDDataset,
        res_per_batch: Optional[int] = None,
        batches_per_epoch: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.re_indices = [
            re_idx
            for re_idx, indices in dataset.flat_indices_by_re_idx.items()
            if len(indices) > 0
        ]
        if not self.re_indices:
            raise ValueError("BalancedReBatchSampler received an empty dataset.")
        self.res_per_batch = res_per_batch or len(self.re_indices)
        self.res_per_batch = min(self.res_per_batch, len(self.re_indices))
        self.shuffle = shuffle
        self.seed = seed
        if batches_per_epoch is None:
            total = sum(
                len(dataset.flat_indices_by_re_idx[re_idx]) for re_idx in self.re_indices
            )
            batches_per_epoch = math.ceil(total / self.res_per_batch)
        self.batches_per_epoch = int(batches_per_epoch)
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        pools = {}
        cursors = {}
        for re_idx in self.re_indices:
            values = np.array(self.dataset.flat_indices_by_re_idx[re_idx], dtype=np.int64)
            if self.shuffle:
                rng.shuffle(values)
            pools[re_idx] = values
            cursors[re_idx] = 0

        re_order = np.array(self.re_indices, dtype=np.int64)
        if self.shuffle:
            rng.shuffle(re_order)

        for batch_idx in range(self.batches_per_epoch):
            if batch_idx % max(1, len(re_order) // self.res_per_batch) == 0 and self.shuffle:
                rng.shuffle(re_order)
            start = (batch_idx * self.res_per_batch) % len(re_order)
            selected_re = np.roll(re_order, -start)[: self.res_per_batch]

            batch = []
            for re_idx in selected_re:
                pool = pools[int(re_idx)]
                cursor = cursors[int(re_idx)]
                if cursor >= len(pool):
                    cursor = 0
                    if self.shuffle:
                        rng.shuffle(pool)
                batch.append(int(pool[cursor]))
                cursors[int(re_idx)] = cursor + 1
            yield batch
        self.epoch += 1

    def __len__(self):
        return self.batches_per_epoch


def _resolve_multi_re_paths(dataset_params):
    data_root = dataset_params.get("data_root") or dataset_params.get("data_dir")
    if not data_root:
        raise ValueError("Multi-Re dataset requires dataset_params.data_root or data_dir.")

    data_file = dataset_params.get("data_file", "mhd_data_3channel.npy")
    re_values = dataset_params.get("re_values")
    if not re_values:
        raise ValueError("Multi-Re dataset requires dataset_params.re_values.")

    paths = []
    for re in re_values:
        template = dataset_params.get("data_dir_template")
        if template:
            directory = template.format(re=re, Re=re)
            if not os.path.isabs(directory):
                directory = os.path.join(data_root, directory)
        else:
            n_value = dataset_params.get("n", dataset_params.get("N", 1000))
            directory = os.path.join(data_root, f"mhd_Re{re}_N{n_value}")
        paths.append(os.path.join(directory, data_file))
    return paths


def get_multi_re_dataloaders(
    dataset_params,
    normalization_config=None,
    batch_size=1,
    num_workers=4,
    seed=42,
):
    normalization_config = _load_normalization_config(normalization_config)
    if normalization_config and "normalization_params" in normalization_config:
        norm_type = normalization_config["normalization_params"].get("type", "unknown")
        print(f"Using {norm_type} normalization at dataset level\n")

    re_values = dataset_params["re_values"]
    rem_values = dataset_params.get("rem_values", re_values)
    data_paths = _resolve_multi_re_paths(dataset_params)

    arrays = []
    for re, path in zip(re_values, data_paths):
        print(f"Loading Re={re} data from {path}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Multi-Re data file not found: {path}")
        raw = np.load(path, mmap_mode="r")
        print(f"  raw shape: {raw.shape}")
        arrays.append(raw)

    common_kwargs = dict(
        datasets=arrays,
        re_values=re_values,
        rem_values=rem_values,
        normalization_config=normalization_config,
        train_size_per_re=dataset_params.get(
            "train_size_per_re", dataset_params.get("train_size")
        ),
        val_plus_test_size_per_re=dataset_params.get(
            "val_plus_test_size_per_re", dataset_params.get("val_plus_test_size")
        ),
        seed=seed,
        sub_t=dataset_params.get("sub_t", 1),
        sub_x=dataset_params.get("sub_x", 1),
        t_range=tuple(dataset_params.get("t_range", (0, 1.0))),
        x_range=tuple(dataset_params.get("x_range", (0, 1.0))),
        y_range=tuple(dataset_params.get("y_range", (0, 1.0))),
    )

    train_dataset = MultiReMHDDataset(split="train", **common_kwargs)
    val_dataset = MultiReMHDDataset(split="val", **common_kwargs)
    test_dataset = MultiReMHDDataset(split="test", **common_kwargs)

    balanced = dataset_params.get("balanced_re_batches", False)
    res_per_batch = dataset_params.get("res_per_batch")
    batches_per_epoch = dataset_params.get("batches_per_epoch")

    if balanced:
        train_sampler = BalancedReBatchSampler(
            train_dataset,
            res_per_batch=res_per_batch,
            batches_per_epoch=batches_per_epoch,
            shuffle=True,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader, test_loader
