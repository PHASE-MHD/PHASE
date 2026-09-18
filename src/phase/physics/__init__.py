"""Differentiable physical operators and residuals."""

from .constraints import compute_constraint_loss, compute_constraints
from .pde_solvers import (
    compute_bfield_pde_loss,
    compute_mhd_bfield_pde,
    compute_mhd_pde,
    compute_pde_loss,
)

__all__ = [
    "compute_constraint_loss",
    "compute_constraints",
    "compute_bfield_pde_loss",
    "compute_mhd_bfield_pde",
    "compute_mhd_pde",
    "compute_pde_loss",
]
