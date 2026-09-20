"""Tests for canonical single-Re, multi-Re, and diffusion data paths."""

import numpy as np
import torch

from phase.data.diffusion_dataset import (
    DiffusionDataset,
    get_diffusion_test_dataloader,
)
from phase.data.multi_re_neurops_dataset import (
    BalancedReBatchSampler,
    MultiReMHDDataset,
)
from phase.data.neurops_dataset import MHDDataset


def test_single_re_splits_are_disjoint():
    data = torch.randn(8, 4, 3, 4, 4)
    initial = data[:, :, :1]
    common = dict(
        x_data=initial,
        y_data=data,
        train_size=4,
        val_plus_test_size=4,
        seed=7,
    )
    train = MHDDataset(split="train", **common)
    val = MHDDataset(split="val", **common)
    test = MHDDataset(split="test", **common)

    assert set(train.indices).isdisjoint(val.indices)
    assert set(train.indices).isdisjoint(test.indices)
    assert set(val.indices).isdisjoint(test.indices)
    inputs, outputs = train[0]
    assert inputs.shape == (7, 3, 4, 4)
    assert outputs.shape == (4, 3, 4, 4)


def test_single_re_sample_fractions_preserve_canonical_split_boundaries():
    data = torch.randn(20, 4, 3, 4, 4)
    common = dict(
        x_data=data[:, :, :1],
        y_data=data,
        train_size=12,
        val_plus_test_size=8,
        seed=11,
    )
    full = {
        split: MHDDataset(split=split, **common)
        for split in ("train", "val", "test")
    }
    reduced = {
        "train": MHDDataset(split="train", sample_fraction=0.5, **common),
        "val": MHDDataset(split="val", sample_fraction=0.5, **common),
        "test": MHDDataset(split="test", sample_fraction=0.5, **common),
    }
    assert [len(reduced[name]) for name in ("train", "val", "test")] == [6, 2, 2]
    for name in reduced:
        assert np.array_equal(
            reduced[name].indices, full[name].indices[: len(reduced[name])]
        )
    assert set(reduced["train"].indices).isdisjoint(reduced["val"].indices)
    assert set(reduced["train"].indices).isdisjoint(reduced["test"].indices)
    assert set(reduced["val"].indices).isdisjoint(reduced["test"].indices)


def test_multi_re_metadata_and_balanced_batches():
    arrays = [
        np.zeros((6, 3, 4, 4, 4), dtype=np.float32),
        np.ones((6, 3, 4, 4, 4), dtype=np.float32),
    ]
    dataset = MultiReMHDDataset(
        datasets=arrays,
        re_values=[80.0, 1000.0],
        rem_values=[80.0, 1000.0],
        split="train",
        train_size_per_re=4,
        val_plus_test_size_per_re=2,
        seed=3,
    )
    inputs, outputs, metadata = dataset[0]
    assert inputs.shape == (7, 3, 4, 4)
    assert outputs.shape == (4, 3, 4, 4)
    assert metadata["nu"] == 1.0 / metadata["re"]
    assert metadata["eta"] == 1.0 / metadata["rem"]

    sampler = BalancedReBatchSampler(
        dataset, res_per_batch=2, batches_per_epoch=3, shuffle=False
    )
    for batch in sampler:
        re_indices = {int(dataset.samples[index][0]) for index in batch}
        assert re_indices == {0, 1}


def test_multi_re_sample_fraction_is_applied_independently_per_regime():
    arrays = [
        np.zeros((10, 3, 4, 4, 4), dtype=np.float32),
        np.ones((10, 3, 4, 4, 4), dtype=np.float32),
    ]
    common = dict(
        datasets=arrays,
        re_values=[80.0, 1000.0],
        rem_values=[80.0, 1000.0],
        train_size_per_re=6,
        val_plus_test_size_per_re=4,
        seed=5,
    )
    full = MultiReMHDDataset(split="train", **common)
    reduced = MultiReMHDDataset(split="train", sample_fraction=0.5, **common)
    assert len(reduced) == 6
    assert all(len(indices) == 3 for indices in reduced.flat_indices_by_re_idx.values())
    for re_idx in (0, 1):
        reduced_samples = [
            reduced.samples[index][1]
            for index in reduced.flat_indices_by_re_idx[re_idx]
        ]
        full_samples = [
            full.samples[index][1]
            for index in full.flat_indices_by_re_idx[re_idx]
        ]
        assert reduced_samples == full_samples[:3]


def test_diffusion_residual_targets_preserve_metadata(tmp_path):
    inputs = np.zeros((2, 4, 3, 4, 4), dtype=np.float32)
    targets = inputs + 2.0
    path = tmp_path / "features.npy"
    np.save(
        path,
        {
            "diff_inputs": inputs,
            "diff_targets": targets,
            "re": np.array([80.0, 1000.0], dtype=np.float32),
            "sample_id": np.array([11, 22], dtype=np.int64),
        },
        allow_pickle=True,
    )
    config = {
        "model_params": {"channels": 4},
        "dataset_params": {
            "prediction_mode": "residual",
            "return_metadata": True,
        },
    }
    dataset = DiffusionDataset(path, config=config, split_name="test")
    x, residual, metadata = dataset[0]
    assert x.shape == (4, 4, 4)
    assert torch.allclose(residual, torch.full_like(residual, 2.0))
    assert metadata["re"].item() == 80.0
    assert metadata["sample_id"].item() == 11


def test_diffusion_evaluation_loader_builds_only_trajectory_test_store(tmp_path):
    test_path = tmp_path / "test"
    test_path.mkdir()
    inputs = np.zeros((2, 4, 3, 4, 4), dtype=np.float32)
    np.save(test_path / "diff_inputs.npy", inputs)
    np.save(test_path / "diff_targets.npy", inputs)
    np.save(test_path / "sample_id.npy", np.array([11, 22], dtype=np.int64))
    config = {
        "model_params": {"channels": 4},
        "normalization_params": {"type": "identity"},
        "dataset_params": {
            "test_path": str(test_path),
            "return_metadata": True,
        },
    }
    loader = get_diffusion_test_dataloader(config, num_workers=0)
    assert loader.batch_size == 3
    assert len(loader.dataset) == 6
    _, _, metadata = next(iter(loader))
    assert torch.equal(metadata["sample_id"], torch.full((3,), 11))


def test_diffusion_evaluation_rejects_flattened_test_store(tmp_path):
    test_path = tmp_path / "flat_test"
    test_path.mkdir()
    inputs = np.zeros((6, 4, 4, 4), dtype=np.float32)
    np.save(test_path / "diff_inputs.npy", inputs)
    np.save(test_path / "diff_targets.npy", inputs)
    config = {
        "model_params": {"channels": 4},
        "normalization_params": {"type": "identity"},
        "dataset_params": {"test_path": str(test_path)},
    }
    try:
        get_diffusion_test_dataloader(config, num_workers=0)
    except ValueError as error:
        assert "trajectory-shaped 5D" in str(error)
    else:
        raise AssertionError("Flattened diffusion evaluation store was accepted.")
