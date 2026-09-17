"""Conversion utilities for canonical PHASE trajectory arrays."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


REPRESENTATIONS = {
    "vector_potential": ("vector potential", 3),
    "direct_b": ("magnetic field", 4),
}


def discover_dedalus_files(input_root: str | Path) -> list[tuple[int, Path]]:
    """Return one HDF5 file per output-N directory, ordered by simulation ID."""
    root = Path(input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input root not found: {root}")

    pattern = re.compile(r"output-(\d+)$")
    found: list[tuple[int, Path]] = []
    for directory in sorted(root.glob("output-*")):
        match = pattern.fullmatch(directory.name)
        if not directory.is_dir() or match is None:
            continue
        files = sorted(directory.glob("*.h5"))
        if not files:
            continue
        if len(files) != 1:
            raise ValueError(
                f"Expected one HDF5 file in {directory}, found {len(files)}"
            )
        found.append((int(match.group(1)), files[0]))
    return sorted(found)


def _validate_tasks(
    path: Path, representation: str
) -> tuple[int, int, int]:
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown representation {representation!r}; "
            f"choose from {sorted(REPRESENTATIONS)}"
        )
    magnetic_task, _ = REPRESENTATIONS[representation]
    with h5py.File(path, "r") as h5:
        velocity = h5["tasks"]["velocity"]
        magnetic = h5["tasks"][magnetic_task]
        if velocity.ndim != 4 or velocity.shape[1] != 2:
            raise ValueError(
                f"{path}: expected velocity shape (T,2,H,W), got {velocity.shape}"
            )
        expected = (
            (velocity.shape[0], velocity.shape[2], velocity.shape[3])
            if representation == "vector_potential"
            else velocity.shape
        )
        if magnetic.shape != expected:
            raise ValueError(
                f"{path}: {magnetic_task} shape {magnetic.shape}, expected {expected}"
            )
        return int(velocity.shape[0]), int(velocity.shape[2]), int(velocity.shape[3])


def convert_dedalus_h5(
    input_root: str | Path,
    output: str | Path,
    *,
    representation: str,
    max_sims: int | None = None,
    dtype: str | np.dtype = "float32",
    overwrite: bool = False,
) -> Path:
    """Convert Dedalus outputs to [sample,time,x,y,channel] NPY format."""
    output = Path(output)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output}")

    files = discover_dedalus_files(input_root)
    if max_sims is not None:
        files = files[:max_sims]
    if not files:
        raise RuntimeError(f"No Dedalus HDF5 files found under {input_root}")

    timesteps, height, width = _validate_tasks(files[0][1], representation)
    magnetic_task, channels = REPRESENTATIONS[representation]
    output.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(len(files), timesteps, height, width, channels),
    )

    expected_velocity = (timesteps, 2, height, width)
    expected_magnetic = (
        (timesteps, height, width)
        if representation == "vector_potential"
        else expected_velocity
    )
    for out_idx, (_sim_idx, path) in enumerate(files):
        with h5py.File(path, "r") as h5:
            velocity = h5["tasks"]["velocity"]
            magnetic = h5["tasks"][magnetic_task]
            if velocity.shape != expected_velocity or magnetic.shape != expected_magnetic:
                raise ValueError(f"{path}: inconsistent task shapes")
            out[out_idx, ..., 0] = velocity[:, 0]
            out[out_idx, ..., 1] = velocity[:, 1]
            if representation == "vector_potential":
                out[out_idx, ..., 2] = magnetic
            else:
                out[out_idx, ..., 2] = magnetic[:, 0]
                out[out_idx, ..., 3] = magnetic[:, 1]
    out.flush()
    return output


def spectral_b_from_a(
    vector_potential: np.ndarray, lx: float = 1.0, ly: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return Bx=dA/dy and By=-dA/dx on a periodic rectangular domain."""
    if vector_potential.ndim < 2:
        raise ValueError("Vector potential must have at least two spatial dimensions")
    nx, ny = vector_potential.shape[-2:]
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kx = kx.reshape((1,) * (vector_potential.ndim - 2) + (nx, 1))
    ky = ky.reshape((1,) * (vector_potential.ndim - 2) + (1, ny))
    a_hat = np.fft.fftn(vector_potential, axes=(-2, -1))
    bx = np.fft.ifftn(1j * ky * a_hat, axes=(-2, -1)).real
    by = np.fft.ifftn(-1j * kx * a_hat, axes=(-2, -1)).real
    return bx, by


def convert_vector_potential_npy(
    input_path: str | Path,
    output_path: str | Path,
    *,
    lx: float = 1.0,
    ly: float = 1.0,
    max_sims: int | None = None,
    chunk_size: int = 8,
    dtype: str | np.dtype = "float32",
    overwrite: bool = False,
) -> Path:
    """Convert [ux,uy,A] trajectories to [ux,uy,Bx,By]."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    data = np.load(input_path, mmap_mode="r")
    if data.ndim != 5 or data.shape[-1] != 3:
        raise ValueError(f"Expected (N,T,X,Y,3), got {data.shape}")
    nsims = data.shape[0] if max_sims is None else min(max_sims, data.shape[0])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.dtype(dtype),
        shape=(nsims, *data.shape[1:4], 4),
    )

    for start in range(0, nsims, chunk_size):
        stop = min(start + chunk_size, nsims)
        chunk = np.asarray(data[start:stop], dtype=np.float64)
        bx, by = spectral_b_from_a(chunk[..., 2], lx=lx, ly=ly)
        out[start:stop, ..., :2] = chunk[..., :2]
        out[start:stop, ..., 2] = bx
        out[start:stop, ..., 3] = by
    out.flush()
    return output_path
