"""Data conversion and train-only statistics for PHASE."""

from .statistics import (
    diffusion_statistics,
    diffusion_statistics_by_re,
    multi_re_magnetic_p99,
    save_npz_statistics,
    split_indices,
    trajectory_statistics,
)
from .conversion import (
    convert_dedalus_h5,
    convert_vector_potential_npy,
    discover_dedalus_files,
    spectral_b_from_a,
)

__all__ = [
    "convert_dedalus_h5",
    "convert_vector_potential_npy",
    "discover_dedalus_files",
    "spectral_b_from_a",
    "diffusion_statistics",
    "diffusion_statistics_by_re",
    "multi_re_magnetic_p99",
    "save_npz_statistics",
    "split_indices",
    "trajectory_statistics",
]
