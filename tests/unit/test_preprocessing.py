"""Synthetic regression tests for PHASE preprocessing."""

from pathlib import Path

import h5py
import numpy as np

from phase.preprocessing import (
    convert_dedalus_h5,
    convert_vector_potential_npy,
    diffusion_statistics,
    multi_re_magnetic_p99,
    spectral_b_from_a,
    split_indices,
    trajectory_statistics,
)


def _write_dedalus_fixture(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    directory = root / "output-0002"
    directory.mkdir(parents=True)
    path = directory / "output-0002_s1.h5"
    nt, nx, ny = 2, 8, 8
    x = np.arange(nx) / nx
    y = np.arange(ny) / ny
    xx, yy = np.meshgrid(x, y, indexing="ij")
    potential = np.stack(
        [np.sin(2 * np.pi * xx) + np.cos(2 * np.pi * yy)] * nt
    )
    bx, by = spectral_b_from_a(potential)
    velocity = np.zeros((nt, 2, nx, ny), dtype=np.float64)
    velocity[:, 0] = 1.0
    velocity[:, 1] = -2.0
    magnetic = np.stack([bx, by], axis=1)
    with h5py.File(path, "w") as h5:
        tasks = h5.create_group("tasks")
        tasks.create_dataset("velocity", data=velocity)
        tasks.create_dataset("vector potential", data=potential)
        tasks.create_dataset("magnetic field", data=magnetic)
    return velocity, potential, magnetic


def test_dedalus_conversion_and_spectral_vector_potential(tmp_path):
    velocity, potential, magnetic = _write_dedalus_fixture(tmp_path / "raw")

    vec_path = tmp_path / "vec.npy"
    convert_dedalus_h5(
        tmp_path / "raw",
        vec_path,
        representation="vector_potential",
    )
    vec = np.load(vec_path)
    assert vec.shape == (1, 2, 8, 8, 3)
    assert np.allclose(vec[0, ..., 0], velocity[:, 0])
    assert np.allclose(vec[0, ..., 1], velocity[:, 1])
    assert np.allclose(vec[0, ..., 2], potential)

    b_path = tmp_path / "b.npy"
    convert_vector_potential_npy(vec_path, b_path)
    direct = np.load(b_path)
    assert np.allclose(direct[0, ..., 2], magnetic[:, 0], atol=1.0e-5)
    assert np.allclose(direct[0, ..., 3], magnetic[:, 1], atol=1.0e-5)

    direct_h5_path = tmp_path / "direct_h5.npy"
    convert_dedalus_h5(
        tmp_path / "raw",
        direct_h5_path,
        representation="direct_b",
    )
    assert np.allclose(np.load(direct_h5_path), direct, atol=1.0e-5)


def test_trajectory_statistics_use_only_deterministic_training_split(tmp_path):
    data = np.zeros((6, 2, 2, 2, 4), dtype=np.float32)
    for sample in range(data.shape[0]):
        data[sample] = sample
    path = tmp_path / "data.npy"
    np.save(path, data)

    indices = split_indices(6, 3, seed=42)
    stats = trajectory_statistics(path, train_size=3, seed=42, chunk_size=2)
    expected = data[indices].mean(axis=(0, 1, 2, 3))
    assert np.array_equal(stats["train_indices"], indices)
    assert np.allclose(stats["mean"], expected)


def test_diffusion_statistics_residual_mode(tmp_path):
    inputs = np.ones((3, 4, 2, 2, 2), dtype=np.float32)
    targets = inputs + np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1, 1)
    path = tmp_path / "features.npy"
    np.save(path, {"diff_inputs": inputs, "diff_targets": targets})
    stats = diffusion_statistics(path, prediction_mode="residual", chunk_size=2)
    assert np.allclose(stats["inputs"]["mean"], np.ones(4))
    assert np.allclose(stats["targets"]["mean"], np.arange(4))


def test_multi_re_p99_reports_paired_scale(tmp_path):
    paths = {}
    for re_value, scale in ((80.0, 1.0), (1000.0, 3.0)):
        data = np.zeros((4, 2, 2, 2, 4), dtype=np.float32)
        data[..., 2] = scale
        data[..., 3] = 0.5 * scale
        path = tmp_path / f"Re{int(re_value)}.npy"
        np.save(path, data)
        paths[re_value] = path

    result = multi_re_magnetic_p99(
        paths,
        train_size=2,
        bins=100,
        chunk_size=1,
    )
    for values in result["per_re"].values():
        assert values["chosen_train_p99_abs_B"] == max(
            values["Bx_p99_abs"], values["By_p99_abs"]
        )
    assert result["global"]["chosen_train_p99_abs_B"] > 2.9


def test_multi_re_p99_default_matches_multi_re_dataset_splits(tmp_path):
    paths = {}
    expected_indices = {}
    for re_index, re_value in enumerate((80.0, 1000.0)):
        data = np.zeros((10, 1, 1, 1, 4), dtype=np.float32)
        data[..., 2] = np.arange(10, dtype=np.float32).reshape(10, 1, 1, 1)
        path = tmp_path / f"Re{int(re_value)}.npy"
        np.save(path, data)
        paths[re_value] = path
        expected_indices[str(re_value)] = np.random.default_rng(
            42 + re_index
        ).permutation(10)[:6]

    result = multi_re_magnetic_p99(
        paths,
        train_size=6,
        bins=100,
        chunk_size=2,
    )

    assert result["settings"]["split_mode"] == "multi_re_seed_plus_index"
    for re_value, expected in expected_indices.items():
        assert np.array_equal(result["per_re"][re_value]["train_indices"], expected)
