"""Utilities for Fourier-space operations in physics-informed neural networks."""

from typing import Tuple, List, Dict, Optional, Any, Union
import torch
import numpy as np


def create_wavenumbers(
    nx: int, ny: int, Lx: float, Ly: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create wavenumbers for Fourier transforms with proper handling of odd/even grid sizes.

    Args:
        nx: Number of grid points in x direction
        ny: Number of grid points in y direction
        Lx: Domain length in x direction
        Ly: Domain length in y direction
        device: Device to place tensors on

    Returns:
        Tuple of wavenumbers (k_x, k_y) reshaped for grid operations
    """
    # Handle x-direction wavenumbers
    k_max_x = nx // 2
    if nx % 2 == 0:  # even size
        # For even sizes, we use -nx/2 to nx/2-1
        k_x_arr = torch.cat(
            [
                torch.arange(start=0, end=k_max_x, step=1, device=device),
                torch.arange(start=-k_max_x, end=0, step=1, device=device),
            ],
            0,
        )
    else:  # odd size
        # For odd sizes, we use -(nx-1)/2 to (nx-1)/2
        k_x_arr = torch.cat(
            [
                torch.arange(start=0, end=k_max_x + 1, step=1, device=device),
                torch.arange(start=-k_max_x, end=0, step=1, device=device),
            ],
            0,
        )

    # Handle y-direction wavenumbers
    k_max_y = ny // 2
    if ny % 2 == 0:  # even size
        k_y_arr = torch.cat(
            [
                torch.arange(start=0, end=ny // 2, step=1, device=device),
                torch.arange(start=-ny // 2, end=0, step=1, device=device),
            ],
            0,
        )
    else:  # odd size
        k_y_arr = torch.cat(
            [
                torch.arange(start=0, end=ny // 2 + 1, step=1, device=device),
                torch.arange(start=-ny // 2, end=0, step=1, device=device),
            ],
            0,
        )

    # Reshape to match grid dimensions (batch, time, x, y)
    k_x = (2 * np.pi / Lx) * k_x_arr.reshape(nx, 1).repeat(1, ny).reshape(1, 1, nx, ny)
    k_y = (2 * np.pi / Ly) * k_y_arr.reshape(1, ny).repeat(nx, 1).reshape(1, 1, nx, ny)

    return k_x, k_y


def compute_derivative(u_h: torch.Tensor, k_i: torch.Tensor) -> torch.Tensor:
    """
    Calculate spatial derivative in Fourier space.

    Args:
        u_h: Fourier transform of the field
        k_i: Wavenumber in the direction of the derivative

    Returns:
        Fourier transform of the derivative
    """
    return 1j * k_i * u_h


def compute_laplacian(
    u_h: torch.Tensor, k_x: torch.Tensor, k_y: torch.Tensor
) -> torch.Tensor:
    """
    Calculate Laplacian in Fourier space.

    Args:
        u_h: Fourier transform of the field
        k_x: Wavenumbers in x direction
        k_y: Wavenumbers in y direction

    Returns:
        Fourier transform of the Laplacian
    """
    lap = -(k_x**2 + k_y**2)
    u_lap_h = lap * u_h
    return u_lap_h


def compute_time_derivative(u: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Compute central difference time derivative.

    Args:
        u: Field tensor of shape (batch, time, x, y)
        dt: Time step

    Returns:
        Time derivative at interior time steps
    """
    return (u[:, 2:] - u[:, :-2]) / (2 * dt)
