"""Constraint calculations for MHD simulations with vector potential."""

from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F
import numpy as np
from ..utils.fourier_utils import create_wavenumbers, compute_derivative


def compute_constraints(
    u: torch.Tensor, v: torch.Tensor, A: torch.Tensor, Lx: float, Ly: float, tend: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute divergence constraints for velocity field and magnetic field.

    Args:
        u: x-component of velocity field (batch, time, x, y)
        v: y-component of velocity field (batch, time, x, y)
        A: z-component of vector potential (batch, time, x, y)
        Lx: Domain length in x direction
        Ly: Domain length in y direction
        tend: Final simulation time

    Returns:
        Tuple of (div_vel, div_B) representing divergence of velocity and magnetic field
    """
    # Get domain dimensions
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)
    ny = u.size(3)
    device = u.device

    # Create wavenumbers for spectral derivatives
    k_x, k_y = create_wavenumbers(nx, ny, Lx, Ly, device)

    # Compute Fourier transforms
    u_h = torch.fft.fftn(u, dim=[2, 3])
    v_h = torch.fft.fftn(v, dim=[2, 3])
    A_h = torch.fft.fftn(A, dim=[2, 3])

    # Compute velocity derivatives
    ux_h = compute_derivative(u_h, k_x)
    vy_h = compute_derivative(v_h, k_y)

    # Compute vector potential derivatives
    Ax_h = compute_derivative(A_h, k_x)
    Ay_h = compute_derivative(A_h, k_y)

    # Magnetic field components (B = ∇ × A)
    Bx_h = Ay_h  # Bx = ∂A/∂y
    By_h = -Ax_h  # By = -∂A/∂x

    # Divergence of B (∇·B = ∂Bx/∂x + ∂By/∂y)
    Bx_x_h = compute_derivative(Bx_h, k_x)
    By_y_h = compute_derivative(By_h, k_y)

    # Transform back to physical space
    ux = torch.fft.irfftn(ux_h, dim=[2, 3], s=(nx, ny))
    vy = torch.fft.irfftn(vy_h, dim=[2, 3], s=(nx, ny))
    Bx_x = torch.fft.irfftn(Bx_x_h, dim=[2, 3], s=(nx, ny))
    By_y = torch.fft.irfftn(By_y_h, dim=[2, 3], s=(nx, ny))

    # Divergence of velocity (∇·v = ∂u/∂x + ∂v/∂y)
    div_vel = ux + vy

    # Divergence of magnetic field (∇·B = ∂Bx/∂x + ∂By/∂y)
    div_B = Bx_x + By_y

    return div_vel, div_B


def compute_constraint_loss(
    div_vel: torch.Tensor,
    div_B: torch.Tensor,
    div_vel_weight: float = 1.0,
    div_B_weight: float = 1.0,
    use_weighted_mean: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute constraint loss for divergence of velocity and magnetic field.

    Args:
        div_vel: Divergence of velocity field
        div_B: Divergence of magnetic field
        div_vel_weight: Weight for velocity divergence term
        div_B_weight: Weight for magnetic field divergence term
        use_weighted_mean: Whether to use weighted mean for loss calculation

    Returns:
        Total constraint loss and dictionary with individual losses
    """
    # Target zero divergence for both fields
    div_vel_val = torch.zeros_like(div_vel)
    div_B_val = torch.zeros_like(div_B)

    # Compute MSE loss for each constraint
    loss_div_vel = F.mse_loss(div_vel, div_vel_val)
    loss_div_B = F.mse_loss(div_B, div_B_val)

    # Compute weighted sum
    weight_sum = (div_vel_weight + div_B_weight) if use_weighted_mean else 1.0
    loss_constraint = (
        div_vel_weight * loss_div_vel + div_B_weight * loss_div_B
    ) / weight_sum

    # Return total loss and individual components
    return loss_constraint, {"div_vel": loss_div_vel.item(), "div_B": loss_div_B.item()}
