"""Utilities for Fourier-space operations in physics-informed neural networks."""

from typing import Tuple
import torch


def create_wavenumbers(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return angular Fourier wavenumbers in FFT storage order."""
    dtype = dtype or torch.get_default_dtype()
    k_x_arr = 2 * torch.pi * torch.fft.fftfreq(
        nx, d=Lx / nx, device=device, dtype=dtype
    )
    k_y_arr = 2 * torch.pi * torch.fft.fftfreq(
        ny, d=Ly / ny, device=device, dtype=dtype
    )
    k_x = k_x_arr.reshape(1, 1, nx, 1).expand(1, 1, nx, ny)
    k_y = k_y_arr.reshape(1, 1, 1, ny).expand(1, 1, nx, ny)
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
