"""PDE calculations for MHD simulations with vector potential."""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F
import numpy as np
from ..utils.fourier_utils import (
    create_wavenumbers,
    compute_derivative,
    compute_laplacian,
    compute_time_derivative,
)


def compute_mhd_pde(
    u: torch.Tensor,
    v: torch.Tensor,
    A: torch.Tensor,
    Lx: float,
    Ly: float,
    tend: float,
    nu: float,
    eta: float,
    rho0: float,
    p: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute PDE residuals for MHD equations using vector potential.

    Args:
        u: x-component of velocity field (batch, time, x, y)
        v: y-component of velocity field (batch, time, x, y)
        A: z-component of vector potential (batch, time, x, y)
        Lx: Domain length in x direction
        Ly: Domain length in y direction
        tend: Final simulation time
        nu: Kinematic viscosity
        eta: Magnetic diffusivity
        rho0: Fluid density
        p: Pressure field (optional)

    Returns:
        Tuple of PDE residuals (Du, Dv, DA)
    """
    # Get domain dimensions
    batchsize = u.size(0)
    nt = u.size(1)
    nx = u.size(2)
    ny = u.size(3)
    device = u.device
    dt = tend / (nt - 1)

    # Create wavenumbers for spectral derivatives
    k_x, k_y = create_wavenumbers(nx, ny, Lx, Ly, device)

    # Compute Laplacian operator for pressure calculation
    lap = -(k_x**2 + k_y**2)
    lap[..., 0, 0] = -1.0  # Avoid division by zero at zero frequency

    # Compute Fourier transforms
    u_h = torch.fft.fftn(u, dim=[2, 3])
    v_h = torch.fft.fftn(v, dim=[2, 3])
    A_h = torch.fft.fftn(A, dim=[2, 3])

    # Compute velocity derivatives
    ux_h = compute_derivative(u_h, k_x)
    uy_h = compute_derivative(u_h, k_y)
    vx_h = compute_derivative(v_h, k_x)
    vy_h = compute_derivative(v_h, k_y)

    # Compute vector potential derivatives
    Ax_h = compute_derivative(A_h, k_x)
    Ay_h = compute_derivative(A_h, k_y)

    # Magnetic field components (B = ∇ × A)
    Bx_h = Ay_h  # Bx = ∂A/∂y
    By_h = -Ax_h  # By = -∂A/∂x
    B2_h = Bx_h**2 + By_h**2

    # More derivatives of magnetic field
    Bx_x_h = compute_derivative(Bx_h, k_x)
    Bx_y_h = compute_derivative(Bx_h, k_y)
    By_x_h = compute_derivative(By_h, k_x)
    By_y_h = compute_derivative(By_h, k_y)

    # Compute Laplacians for diffusion terms
    u_lap_h = compute_laplacian(u_h, k_x, k_y)
    v_lap_h = compute_laplacian(v_h, k_x, k_y)
    A_lap_h = compute_laplacian(A_h, k_x, k_y)

    # Transform all fields back to physical space
    ux = torch.fft.irfftn(ux_h, dim=[2, 3], s=(nx, ny))
    uy = torch.fft.irfftn(uy_h, dim=[2, 3], s=(nx, ny))
    vx = torch.fft.irfftn(vx_h, dim=[2, 3], s=(nx, ny))
    vy = torch.fft.irfftn(vy_h, dim=[2, 3], s=(nx, ny))
    Ax = torch.fft.irfftn(Ax_h, dim=[2, 3], s=(nx, ny))
    Ay = torch.fft.irfftn(Ay_h, dim=[2, 3], s=(nx, ny))
    Bx = torch.fft.irfftn(Bx_h, dim=[2, 3], s=(nx, ny))
    By = torch.fft.irfftn(By_h, dim=[2, 3], s=(nx, ny))
    B2 = torch.fft.irfftn(B2_h, dim=[2, 3], s=(nx, ny))
    Bx_x = torch.fft.irfftn(Bx_x_h, dim=[2, 3], s=(nx, ny))
    Bx_y = torch.fft.irfftn(Bx_y_h, dim=[2, 3], s=(nx, ny))
    By_x = torch.fft.irfftn(By_x_h, dim=[2, 3], s=(nx, ny))
    By_y = torch.fft.irfftn(By_y_h, dim=[2, 3], s=(nx, ny))
    u_lap = torch.fft.irfftn(u_lap_h, dim=[2, 3], s=(nx, ny))
    v_lap = torch.fft.irfftn(v_lap_h, dim=[2, 3], s=(nx, ny))
    A_lap = torch.fft.irfftn(A_lap_h, dim=[2, 3], s=(nx, ny))

    # Calculate pressure
    if p is None:
        # Compute pressure by solving the pressure Poisson equation
        div_vel_grad_vel = ux**2 + 2 * uy * vx + vy**2
        div_B_grad_B = Bx_x**2 + 2 * Bx_y * By_x + By_y**2
        div_vel_grad_vel_h = torch.fft.fftn(div_vel_grad_vel, dim=[2, 3])
        div_B_grad_B_h = torch.fft.fftn(div_B_grad_B, dim=[2, 3])

        # Solve Poisson equation for total pressure
        ptot_h = (div_B_grad_B_h - rho0 * div_vel_grad_vel_h) / lap
        ptot_h[..., 0, 0] = B2_h[..., 0, 0] / 2.0  # Set mean pressure

        # Calculate fluid pressure by subtracting magnetic pressure
        p_h = ptot_h - B2_h / 2.0
    else:
        p_h = torch.fft.fftn(p, dim=[2, 3])
        ptot_h = p_h + B2_h / 2.0

    # Compute pressure gradients
    ptot_x_h = compute_derivative(ptot_h, k_x)
    ptot_y_h = compute_derivative(ptot_h, k_y)

    # Transform pressure fields back to physical space
    p = torch.fft.irfftn(p_h, dim=[2, 3], s=(nx, ny))
    ptot = torch.fft.irfftn(ptot_h, dim=[2, 3], s=(nx, ny))
    ptot_x = torch.fft.irfftn(ptot_x_h, dim=[2, 3], s=(nx, ny))
    ptot_y = torch.fft.irfftn(ptot_y_h, dim=[2, 3], s=(nx, ny))

    # Calculate advection terms
    vel_grad_u = u * ux + v * uy  # (u·∇)u
    vel_grad_v = u * vx + v * vy  # (u·∇)v
    vel_grad_A = u * Ax + v * Ay  # (u·∇)A

    # Calculate Lorentz force terms
    B_grad_Bx = Bx * Bx_x + By * Bx_y  # (B·∇)Bx
    B_grad_By = Bx * By_x + By * By_y  # (B·∇)By

    # Right-hand sides of the MHD equations
    u_rhs = -vel_grad_u - ptot_x / rho0 + B_grad_Bx / rho0 + nu * u_lap
    v_rhs = -vel_grad_v - ptot_y / rho0 + B_grad_By / rho0 + nu * v_lap
    A_rhs = -vel_grad_A + eta * A_lap

    # Compute time derivatives
    u_t = compute_time_derivative(u, dt)
    v_t = compute_time_derivative(v, dt)
    A_t = compute_time_derivative(A, dt)

    # Compute PDE residuals (should be zero for exact solution)
    Du = u_t - u_rhs[:, 1:-1]
    Dv = v_t - v_rhs[:, 1:-1]
    DA = A_t - A_rhs[:, 1:-1]

    return Du, Dv, DA


def compute_pde_loss(
    Du: torch.Tensor,
    Dv: torch.Tensor,
    DA: torch.Tensor,
    Du_weight: float = 1.0,
    Dv_weight: float = 1.0,
    DA_weight: float = 1.0,
    use_weighted_mean: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute PDE loss from residuals.

    Args:
        Du: u-momentum equation residual
        Dv: v-momentum equation residual
        DA: vector potential equation residual
        Du_weight: Weight for u-momentum residual
        Dv_weight: Weight for v-momentum residual
        DA_weight: Weight for vector potential residual
        use_weighted_mean: Whether to use weighted mean for loss calculation

    Returns:
        Total PDE loss and dictionary with individual losses
    """
    # Target zero residual for PDEs
    Du_val = torch.zeros_like(Du)
    Dv_val = torch.zeros_like(Dv)
    DA_val = torch.zeros_like(DA)

    # Compute MSE loss for each PDE residual
    loss_Du = F.mse_loss(Du, Du_val)
    loss_Dv = F.mse_loss(Dv, Dv_val)
    loss_DA = F.mse_loss(DA, DA_val)

    # Compute weighted sum
    weight_sum = (Du_weight + Dv_weight + DA_weight) if use_weighted_mean else 1.0
    loss_pde = (
        Du_weight * loss_Du + Dv_weight * loss_Dv + DA_weight * loss_DA
    ) / weight_sum

    # Return total loss and individual components
    return loss_pde, {"Du": loss_Du.item(), "Dv": loss_Dv.item(), "DA": loss_DA.item()}

def compute_mhd_bfield_pde(
    u: torch.Tensor,
    v: torch.Tensor,
    Bx: torch.Tensor,
    By: torch.Tensor,
    Lx: float,
    Ly: float,
    tend: float,
    nu: float,
    eta: float,
    rho0: float,
    p: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute incompressible MHD residuals for direct magnetic fields [u, v, Bx, By].

    Uses the B-form induction equation:
        dB/dt = -(u . grad) B + (B . grad) u + eta laplacian(B)

    Momentum uses total pressure from the incompressible pressure Poisson equation:
        du/dt = -(u . grad) u - grad(p_tot)/rho0 + (B . grad) B / rho0 + nu laplacian(u)
    """
    nt = u.size(1)
    nx = u.size(2)
    ny = u.size(3)
    device = u.device
    dt = tend / (nt - 1)

    k_x, k_y = create_wavenumbers(nx, ny, Lx, Ly, device)
    lap = -(k_x**2 + k_y**2)
    lap[..., 0, 0] = -1.0

    u_h = torch.fft.fftn(u, dim=[2, 3])
    v_h = torch.fft.fftn(v, dim=[2, 3])
    Bx_h = torch.fft.fftn(Bx, dim=[2, 3])
    By_h = torch.fft.fftn(By, dim=[2, 3])

    ux_h = compute_derivative(u_h, k_x)
    uy_h = compute_derivative(u_h, k_y)
    vx_h = compute_derivative(v_h, k_x)
    vy_h = compute_derivative(v_h, k_y)

    Bx_x_h = compute_derivative(Bx_h, k_x)
    Bx_y_h = compute_derivative(Bx_h, k_y)
    By_x_h = compute_derivative(By_h, k_x)
    By_y_h = compute_derivative(By_h, k_y)

    u_lap_h = compute_laplacian(u_h, k_x, k_y)
    v_lap_h = compute_laplacian(v_h, k_x, k_y)
    Bx_lap_h = compute_laplacian(Bx_h, k_x, k_y)
    By_lap_h = compute_laplacian(By_h, k_x, k_y)

    ux = torch.fft.ifftn(ux_h, dim=[2, 3]).real
    uy = torch.fft.ifftn(uy_h, dim=[2, 3]).real
    vx = torch.fft.ifftn(vx_h, dim=[2, 3]).real
    vy = torch.fft.ifftn(vy_h, dim=[2, 3]).real
    Bx_x = torch.fft.ifftn(Bx_x_h, dim=[2, 3]).real
    Bx_y = torch.fft.ifftn(Bx_y_h, dim=[2, 3]).real
    By_x = torch.fft.ifftn(By_x_h, dim=[2, 3]).real
    By_y = torch.fft.ifftn(By_y_h, dim=[2, 3]).real
    u_lap = torch.fft.ifftn(u_lap_h, dim=[2, 3]).real
    v_lap = torch.fft.ifftn(v_lap_h, dim=[2, 3]).real
    Bx_lap = torch.fft.ifftn(Bx_lap_h, dim=[2, 3]).real
    By_lap = torch.fft.ifftn(By_lap_h, dim=[2, 3]).real

    if p is None:
        div_vel_grad_vel = ux**2 + 2.0 * uy * vx + vy**2
        div_B_grad_B = Bx_x**2 + 2.0 * Bx_y * By_x + By_y**2
        rhs_h = torch.fft.fftn(div_B_grad_B - rho0 * div_vel_grad_vel, dim=[2, 3])
        ptot_h = rhs_h / lap
        ptot_h[..., 0, 0] = 0.0
    else:
        p_h = torch.fft.fftn(p, dim=[2, 3])
        B2_h = torch.fft.fftn(0.5 * (Bx**2 + By**2), dim=[2, 3])
        ptot_h = p_h + B2_h

    ptot_x = torch.fft.ifftn(compute_derivative(ptot_h, k_x), dim=[2, 3]).real
    ptot_y = torch.fft.ifftn(compute_derivative(ptot_h, k_y), dim=[2, 3]).real

    vel_grad_u = u * ux + v * uy
    vel_grad_v = u * vx + v * vy
    vel_grad_Bx = u * Bx_x + v * Bx_y
    vel_grad_By = u * By_x + v * By_y

    B_grad_Bx = Bx * Bx_x + By * Bx_y
    B_grad_By = Bx * By_x + By * By_y
    B_grad_u = Bx * ux + By * uy
    B_grad_v = Bx * vx + By * vy

    u_rhs = -vel_grad_u - ptot_x / rho0 + B_grad_Bx / rho0 + nu * u_lap
    v_rhs = -vel_grad_v - ptot_y / rho0 + B_grad_By / rho0 + nu * v_lap
    Bx_rhs = -vel_grad_Bx + B_grad_u + eta * Bx_lap
    By_rhs = -vel_grad_By + B_grad_v + eta * By_lap

    u_t = compute_time_derivative(u, dt)
    v_t = compute_time_derivative(v, dt)
    Bx_t = compute_time_derivative(Bx, dt)
    By_t = compute_time_derivative(By, dt)

    Du = u_t - u_rhs[:, 1:-1]
    Dv = v_t - v_rhs[:, 1:-1]
    DBx = Bx_t - Bx_rhs[:, 1:-1]
    DBy = By_t - By_rhs[:, 1:-1]

    return Du, Dv, DBx, DBy


def compute_bfield_pde_loss(
    Du: torch.Tensor,
    Dv: torch.Tensor,
    DBx: torch.Tensor,
    DBy: torch.Tensor,
    Du_weight: float = 1.0,
    Dv_weight: float = 1.0,
    DBx_weight: float = 1.0,
    DBy_weight: float = 1.0,
    use_weighted_mean: bool = False,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute MSE loss for direct-B MHD residuals."""
    loss_Du = F.mse_loss(Du, torch.zeros_like(Du))
    loss_Dv = F.mse_loss(Dv, torch.zeros_like(Dv))
    loss_DBx = F.mse_loss(DBx, torch.zeros_like(DBx))
    loss_DBy = F.mse_loss(DBy, torch.zeros_like(DBy))
    weight_sum = (
        Du_weight + Dv_weight + DBx_weight + DBy_weight
        if use_weighted_mean
        else 1.0
    )
    loss_pde = (
        Du_weight * loss_Du
        + Dv_weight * loss_Dv
        + DBx_weight * loss_DBx
        + DBy_weight * loss_DBy
    ) / weight_sum
    return loss_pde, {
        "Du": loss_Du.item(),
        "Dv": loss_Dv.item(),
        "DBx": loss_DBx.item(),
        "DBy": loss_DBy.item(),
    }
