"""Common Fourier-space diagnostics for periodic two-dimensional MHD."""

from __future__ import annotations

import math

import torch


FIELD_NAMES = ("ux", "uy", "Bx", "By", "omega", "j")


def _check_sequence(sequence: torch.Tensor, channels: tuple[int, ...]) -> None:
    if sequence.ndim != 4:
        raise ValueError(
            "Expected one trajectory with shape [channels,time,x,y], got "
            f"{tuple(sequence.shape)}."
        )
    if sequence.shape[0] not in channels:
        raise ValueError(
            f"Expected {channels} channels, got {sequence.shape[0]}."
        )
    if not torch.isfinite(sequence).all():
        raise ValueError("Evaluation trajectory contains non-finite values.")


def wavenumbers(
    nx: int,
    ny: int,
    *,
    lx: float,
    ly: float,
    device,
    dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return angular Fourier wavenumbers along the stored x/y axes."""
    kx = 2.0 * math.pi * torch.fft.fftfreq(
        nx, d=lx / nx, device=device, dtype=dtype
    ).reshape(1, nx, 1)
    ky = 2.0 * math.pi * torch.fft.fftfreq(
        ny, d=ly / ny, device=device, dtype=dtype
    ).reshape(1, 1, ny)
    return kx, ky


def spectral_derivatives(
    field: torch.Tensor, lx: float = 1.0, ly: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiate ``[time,x,y]`` fields using the periodic Fourier basis."""
    if field.ndim != 3:
        raise ValueError(f"Expected [time,x,y], got {tuple(field.shape)}.")
    _, nx, ny = field.shape
    kx, ky = wavenumbers(
        nx, ny, lx=lx, ly=ly, device=field.device, dtype=field.dtype
    )
    field_hat = torch.fft.fft2(field, dim=(-2, -1))
    dx = torch.fft.ifft2(1j * kx * field_hat, dim=(-2, -1)).real
    dy = torch.fft.ifft2(1j * ky * field_hat, dim=(-2, -1)).real
    return dx, dy


def curl_2d(
    qx: torch.Tensor, qy: torch.Tensor, lx: float = 1.0, ly: float = 1.0
) -> torch.Tensor:
    """Return ``d(qy)/dx - d(qx)/dy`` for ``[time,x,y]`` vectors."""
    _, dqx_dy = spectral_derivatives(qx, lx, ly)
    dqy_dx, _ = spectral_derivatives(qy, lx, ly)
    return dqy_dx - dqx_dy


def divergence_2d(
    qx: torch.Tensor, qy: torch.Tensor, lx: float = 1.0, ly: float = 1.0
) -> torch.Tensor:
    """Return ``d(qx)/dx + d(qy)/dy`` for ``[time,x,y]`` vectors."""
    dqx_dx, _ = spectral_derivatives(qx, lx, ly)
    _, dqy_dy = spectral_derivatives(qy, lx, ly)
    return dqx_dx + dqy_dy


def magnetic_field_from_potential(
    potential: torch.Tensor, lx: float = 1.0, ly: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct ``Bx=dA/dy`` and ``By=-dA/dx`` from ``A[time,x,y]``."""
    dA_dx, dA_dy = spectral_derivatives(potential, lx, ly)
    return dA_dy, -dA_dx


def derive_fields(
    sequence: torch.Tensor,
    representation: str,
    lx: float = 1.0,
    ly: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Convert a physical trajectory to the six common evaluation fields."""
    representation = representation.lower()
    expected = (3,) if representation == "vector_potential" else (4,)
    if representation not in {"vector_potential", "direct_b"}:
        raise ValueError("representation must be 'vector_potential' or 'direct_b'.")
    _check_sequence(sequence, expected)

    ux, uy = sequence[0], sequence[1]
    if representation == "vector_potential":
        bx, by = magnetic_field_from_potential(sequence[2], lx, ly)
    else:
        bx, by = sequence[2], sequence[3]
    return {
        "ux": ux,
        "uy": uy,
        "Bx": bx,
        "By": by,
        "omega": curl_2d(ux, uy, lx, ly),
        "j": curl_2d(bx, by, lx, ly),
    }


def _shell_map(nx: int, ny: int, device, dtype) -> torch.Tensor:
    kx = torch.fft.fftfreq(nx, d=1.0 / nx, device=device, dtype=dtype)
    ky = torch.fft.fftfreq(ny, d=1.0 / ny, device=device, dtype=dtype)
    return torch.sqrt(kx.reshape(nx, 1).square() + ky.reshape(1, ny).square())


def scalar_spectrum(field: torch.Tensor) -> torch.Tensor:
    """Return integer-shell power for one ``[x,y]`` scalar field."""
    if field.ndim != 2:
        raise ValueError(f"Expected [x,y], got {tuple(field.shape)}.")
    nx, ny = field.shape
    shell_map = _shell_map(nx, ny, field.device, field.dtype)
    power = torch.abs(torch.fft.fft2(field) / (nx * ny)).square()
    values = []
    for mode in range(1, min(nx, ny) // 2 + 1):
        mask = torch.abs(shell_map - mode) < 0.5
        values.append(power[mask].sum())
    return torch.stack(values)


def vector_spectrum(qx: torch.Tensor, qy: torch.Tensor) -> torch.Tensor:
    """Return integer-shell power for one ``[x,y]`` vector field."""
    return scalar_spectrum(qx) + scalar_spectrum(qy)
