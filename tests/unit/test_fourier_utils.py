"""Regression tests for shared Fourier derivative utilities."""

from __future__ import annotations

import math

import pytest
import torch

from phase.evaluation.physics import spectral_derivatives
from phase.utils.fourier_utils import create_wavenumbers, compute_derivative


@pytest.mark.parametrize(("nx", "ny"), [(30, 42), (31, 41)])
def test_wavenumbers_match_torch_fftfreq_for_even_and_odd_grids(nx, ny):
    lx, ly = 2.0, 3.0
    kx, ky = create_wavenumbers(nx, ny, lx, ly, torch.device("cpu"))
    expected_kx = 2.0 * torch.pi * torch.fft.fftfreq(nx, d=lx / nx)
    expected_ky = 2.0 * torch.pi * torch.fft.fftfreq(ny, d=ly / ny)

    assert kx.shape == (1, 1, nx, ny)
    assert ky.shape == (1, 1, nx, ny)
    assert torch.allclose(kx[0, 0, :, 0], expected_kx)
    assert torch.allclose(ky[0, 0, 0, :], expected_ky)


@pytest.mark.parametrize(("nx", "ny"), [(30, 42), (31, 41)])
def test_shared_and_evaluation_derivatives_match_rectangular_analytic_field(nx, ny):
    lx, ly = 2.0, 3.0
    mode_x, mode_y = 3, 5
    x = torch.arange(nx, dtype=torch.float64) * lx / nx
    y = torch.arange(ny, dtype=torch.float64) * ly / ny
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    kx_value = 2.0 * math.pi * mode_x / lx
    ky_value = 2.0 * math.pi * mode_y / ly
    field = torch.sin(kx_value * xx) * torch.cos(ky_value * yy)
    expected_dx = kx_value * torch.cos(kx_value * xx) * torch.cos(ky_value * yy)
    expected_dy = -ky_value * torch.sin(kx_value * xx) * torch.sin(ky_value * yy)

    eval_dx, eval_dy = spectral_derivatives(field.unsqueeze(0), lx, ly)
    kx, ky = create_wavenumbers(nx, ny, lx, ly, field.device)
    field_hat = torch.fft.fftn(field[None, None], dim=(2, 3))
    shared_dx = torch.fft.ifftn(compute_derivative(field_hat, kx), dim=(2, 3)).real
    shared_dy = torch.fft.ifftn(compute_derivative(field_hat, ky), dim=(2, 3)).real

    assert torch.allclose(eval_dx[0], expected_dx, atol=2.0e-11, rtol=0)
    assert torch.allclose(eval_dy[0], expected_dy, atol=2.0e-11, rtol=0)
    assert torch.allclose(shared_dx[0, 0], expected_dx, atol=2.0e-11, rtol=0)
    assert torch.allclose(shared_dy[0, 0], expected_dy, atol=2.0e-11, rtol=0)
