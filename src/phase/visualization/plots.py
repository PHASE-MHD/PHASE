"""Paper-ready plots built from denormalized evaluation records."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import ScalarFormatter

from phase.evaluation.metrics import EPS, PDF_SPECS
from phase.evaluation.physics import derive_fields, scalar_spectrum, vector_spectrum


FIELD_ORDER = ("ux", "uy", "Bx", "By", "omega", "j")
FIELD_LABELS = {
    "ux": r"$u_x$",
    "uy": r"$u_y$",
    "Bx": r"$B_x$",
    "By": r"$B_y$",
    "omega": r"$\omega$",
    "j": r"$J$",
}
FIELD_CMAPS = {
    "ux": "PuOr_r",
    "uy": "PuOr_r",
    "Bx": "RdBu_r",
    "By": "RdBu_r",
    "omega": "PuOr_r",
    "j": "RdBu_r",
}


def resolve_time_indices(
    nt: int,
    time_range: tuple[float, float],
    requested_times: Iterable[float] | None,
) -> tuple[np.ndarray, list[int]]:
    """Map requested physical times to unique nearest stored frames."""
    if nt <= 0:
        raise ValueError("A trajectory must contain at least one frame.")
    start, stop = (float(value) for value in time_range)
    if not np.isfinite(start) or not np.isfinite(stop) or stop < start:
        raise ValueError(f"Invalid time range {time_range}.")
    times = np.linspace(start, stop, nt)
    if requested_times is None:
        return times, [nt - 1]
    indices = []
    tolerance = 1.0e-9 * max(1.0, abs(start), abs(stop))
    for requested in requested_times:
        requested = float(requested)
        if not np.isfinite(requested):
            raise ValueError("Requested visualization times must be finite.")
        if requested < start - tolerance or requested > stop + tolerance:
            raise ValueError(
                f"Requested time {requested} is outside [{start}, {stop}]."
            )
        index = int(np.argmin(np.abs(times - requested)))
        if index not in indices:
            indices.append(index)
    if not indices:
        raise ValueError("At least one visualization time is required.")
    return times, indices


def _save_figure(fig, output_stem: str | Path, formats, dpi: int) -> list[Path]:
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for extension in formats:
        extension = str(extension).lower().lstrip(".")
        if extension not in {"png", "pdf"}:
            raise ValueError(f"Unsupported figure format: {extension}")
        path = output_stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def _scientific_colorbar(colorbar, fontsize: int) -> None:
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))
    colorbar.formatter = formatter
    colorbar.ax.tick_params(labelsize=fontsize)
    colorbar.update_ticks()


def _imshow(ax, field, *, lx: float = 1.0, ly: float = 1.0, **kwargs):
    # Arrays are stored [x,y]; imshow expects image rows/columns [y,x].
    return ax.imshow(
        field.detach().cpu().numpy().T,
        origin="lower",
        extent=(0.0, lx, 0.0, ly),
        aspect="equal",
        **kwargs,
    )


def plot_field_comparison(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    representation: str,
    *,
    time_index: int,
    time_value: float,
    output_stem: str | Path,
    formats=("png", "pdf"),
    dpi: int = 240,
    lx: float = 1.0,
    ly: float = 1.0,
) -> list[Path]:
    """Plot model, DNS, and DNS-model errors for all six physical fields."""
    pred = derive_fields(prediction, representation, lx, ly)
    true = derive_fields(truth, representation, lx, ly)
    fig, axes = plt.subplots(6, 3, figsize=(9.2, 16.5), squeeze=False)
    for column, title in enumerate(("Model", "DNS", "Error")):
        axes[0, column].set_title(title, fontsize=13)

    for row, name in enumerate(FIELD_ORDER):
        pred_field = pred[name][time_index]
        true_field = true[name][time_index]
        error = true_field - pred_field
        # DNS determines the physical color scale; model saturation remains visible.
        limit = max(float(torch.max(torch.abs(true_field))), 1.0e-12)
        error_limit = max(float(torch.max(torch.abs(error))), 1.0e-12)
        arrays = (pred_field, true_field, error)
        for column, field in enumerate(arrays):
            if column < 2:
                image = _imshow(
                    axes[row, column],
                    field,
                    lx=lx,
                    ly=ly,
                    cmap=FIELD_CMAPS[name],
                    vmin=-limit,
                    vmax=limit,
                )
            else:
                image = _imshow(
                    axes[row, column],
                    field,
                    lx=lx,
                    ly=ly,
                    cmap="viridis",
                    vmin=-error_limit,
                    vmax=error_limit,
                )
            colorbar = fig.colorbar(
                image, ax=axes[row, column], fraction=0.046, pad=0.025
            )
            _scientific_colorbar(colorbar, 8)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(FIELD_LABELS[name], fontsize=13)
    fig.suptitle(f"$t={time_value:g}$", fontsize=14, y=0.995)
    fig.subplots_adjust(wspace=0.10, hspace=0.08, top=0.96)
    return _save_figure(fig, output_stem, formats, dpi)


def _normalized_spectrum(values: torch.Tensor) -> torch.Tensor:
    return values / (torch.sum(values) + EPS)


def plot_spectrum_comparison(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    representation: str,
    *,
    time_index: int,
    time_value: float,
    output_stem: str | Path,
    formats=("png", "pdf"),
    dpi: int = 240,
    lx: float = 1.0,
    ly: float = 1.0,
) -> list[Path]:
    """Plot total-power-normalized spectra for primary and derived fields."""
    pred = derive_fields(prediction, representation, lx, ly)
    true = derive_fields(truth, representation, lx, ly)
    spectra = {
        "Kinetic energy": (
            vector_spectrum(pred["ux"][time_index], pred["uy"][time_index]),
            vector_spectrum(true["ux"][time_index], true["uy"][time_index]),
        ),
        "Magnetic energy": (
            vector_spectrum(pred["Bx"][time_index], pred["By"][time_index]),
            vector_spectrum(true["Bx"][time_index], true["By"][time_index]),
        ),
        "Vorticity": (
            scalar_spectrum(pred["omega"][time_index]),
            scalar_spectrum(true["omega"][time_index]),
        ),
        "Current density": (
            scalar_spectrum(pred["j"][time_index]),
            scalar_spectrum(true["j"][time_index]),
        ),
    }
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), sharey=True)
    for index, (title, (model_spectrum, dns_spectrum)) in enumerate(spectra.items()):
        model_spectrum = _normalized_spectrum(model_spectrum).cpu().numpy()
        dns_spectrum = _normalized_spectrum(dns_spectrum).cpu().numpy()
        k = np.arange(1, len(model_spectrum) + 1)
        axes[index].loglog(k, dns_spectrum, color="black", label="DNS")
        axes[index].loglog(k, model_spectrum, color="#2166ac", label="Model")
        axes[index].set_title(title, fontsize=12)
        axes[index].set_xlabel(r"$k$", fontsize=12)
        axes[index].set_xlim(1, len(k))
        axes[index].tick_params(direction="in", top=True, right=True)
    axes[0].set_ylabel(r"$E(k)/\sum_k E(k)$", fontsize=12)
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle(f"$t={time_value:g}$", fontsize=13)
    fig.subplots_adjust(wspace=0.08, top=0.82)
    return _save_figure(fig, output_stem, formats, dpi)


def _true_rms_scales(fields: dict[str, torch.Tensor], time_index: int):
    ux = fields["ux"][time_index]
    uy = fields["uy"][time_index]
    bx = fields["Bx"][time_index]
    by = fields["By"][time_index]
    return {
        "ux": torch.sqrt(torch.mean(ux.square() + uy.square())) + EPS,
        "uy": torch.sqrt(torch.mean(ux.square() + uy.square())) + EPS,
        "Bx": torch.sqrt(torch.mean(bx.square() + by.square())) + EPS,
        "By": torch.sqrt(torch.mean(bx.square() + by.square())) + EPS,
        "omega": torch.sqrt(torch.mean(fields["omega"][time_index].square())) + EPS,
        "j": torch.sqrt(torch.mean(fields["j"][time_index].square())) + EPS,
    }


def plot_pdf_comparison(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    representation: str,
    *,
    time_index: int,
    time_value: float,
    output_stem: str | Path,
    formats=("png", "pdf"),
    dpi: int = 240,
    lx: float = 1.0,
    ly: float = 1.0,
) -> list[Path]:
    """Plot DNS-RMS-normalized PDFs with the evaluation histogram contract."""
    pred = derive_fields(prediction, representation, lx, ly)
    true = derive_fields(truth, representation, lx, ly)
    scales = _true_rms_scales(true, time_index)
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0), sharey=True)
    for ax, name in zip(axes.ravel(), FIELD_ORDER):
        bins, value_range = PDF_SPECS[name]
        model = (pred[name][time_index] / scales[name]).cpu().numpy().ravel()
        dns = (true[name][time_index] / scales[name]).cpu().numpy().ravel()
        dns_hist, edges = np.histogram(
            dns, bins=bins, range=value_range, density=True
        )
        model_hist, _ = np.histogram(model, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.semilogy(centers, dns_hist, color="black", label="DNS")
        ax.semilogy(centers, model_hist, color="#2166ac", label="Model")
        ax.set_xlim(value_range)
        ax.set_ylim(2.0e-3, 10.0)
        ax.set_xlabel(FIELD_LABELS[name] + "/RMS", fontsize=12)
        ax.tick_params(direction="in", top=True, right=True)
    axes[0, 0].set_ylabel("PDF", fontsize=12)
    axes[1, 0].set_ylabel("PDF", fontsize=12)
    axes[0, 0].legend(frameon=False, fontsize=10)
    fig.suptitle(f"$t={time_value:g}$", fontsize=13)
    fig.subplots_adjust(wspace=0.08, hspace=0.28, top=0.91)
    return _save_figure(fig, output_stem, formats, dpi)


def initial_kh_tracer(
    nx: int,
    ny: int,
    delta: float = 0.05,
    *,
    ly: float = 1.0,
) -> np.ndarray:
    y = np.linspace(0.0, ly, ny, endpoint=False)
    profile = (
        np.tanh((y - 0.25 * ly) / delta)
        - np.tanh((y - 0.75 * ly) / delta)
        - 1.0
    )
    return np.broadcast_to(profile, (nx, ny)).copy()


def _bilinear_periodic(field, x_query, y_query, lx: float, ly: float):
    nx, ny = field.shape
    x = (x_query % lx) * nx / lx
    y = (y_query % ly) * ny / ly
    i0 = np.floor(x).astype(np.int64) % nx
    j0 = np.floor(y).astype(np.int64) % ny
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    wx = x - np.floor(x)
    wy = y - np.floor(y)
    return (
        (1 - wx) * (1 - wy) * field[i0, j0]
        + wx * (1 - wy) * field[i1, j0]
        + (1 - wx) * wy * field[i0, j1]
        + wx * wy * field[i1, j1]
    )


def _diffuse_periodic(
    field,
    diffusivity: float,
    dt: float,
    lx: float,
    ly: float,
):
    if diffusivity <= 0.0 or dt <= 0.0:
        return field
    nx, ny = field.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    damping = np.exp(-diffusivity * (kx[:, None] ** 2 + ky[None, :] ** 2) * dt)
    return np.fft.ifft2(np.fft.fft2(field) * damping).real


def advect_tracer(
    ux: torch.Tensor | np.ndarray,
    uy: torch.Tensor | np.ndarray,
    times: np.ndarray,
    *,
    delta: float = 0.05,
    diffusivity: float = 0.001,
    lx: float = 1.0,
    ly: float = 1.0,
) -> np.ndarray:
    """Post-process a passive KH dye using periodic semi-Lagrangian advection."""
    ux = np.asarray(torch.as_tensor(ux).cpu(), dtype=np.float64)
    uy = np.asarray(torch.as_tensor(uy).cpu(), dtype=np.float64)
    if ux.shape != uy.shape or ux.ndim != 3:
        raise ValueError("Tracer velocities must have matching [time,x,y] shapes.")
    times = np.asarray(times, dtype=np.float64)
    if ux.shape[0] == 0 or ux.shape[1] == 0 or ux.shape[2] == 0:
        raise ValueError("Tracer velocity trajectories must be nonempty.")
    if (
        len(times) != ux.shape[0]
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0.0)
    ):
        raise ValueError("Tracer times must be strictly increasing and match velocity.")
    if not np.isfinite(ux).all() or not np.isfinite(uy).all():
        raise ValueError("Tracer velocities must be finite.")
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("Tracer interface width must be finite and positive.")
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("Tracer diffusivity must be finite and nonnegative.")
    if not np.isfinite(lx) or not np.isfinite(ly) or lx <= 0.0 or ly <= 0.0:
        raise ValueError("Tracer domain lengths must be finite and positive.")
    nt, nx, ny = ux.shape
    x = np.linspace(0.0, lx, nx, endpoint=False)
    y = np.linspace(0.0, ly, ny, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    tracer = np.empty((nt, nx, ny), dtype=np.float32)
    tracer[0] = initial_kh_tracer(nx, ny, delta, ly=ly)
    for index in range(nt - 1):
        dt = float(times[index + 1] - times[index])
        back_x = xx - dt * ux[index]
        back_y = yy - dt * uy[index]
        step = _bilinear_periodic(tracer[index], back_x, back_y, lx, ly)
        tracer[index + 1] = _diffuse_periodic(
            step, diffusivity, dt, lx, ly
        )
    return tracer


def plot_tracer_comparison(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    time_index: int,
    time_value: float,
    output_stem: str | Path,
    formats=("png", "pdf"),
    dpi: int = 240,
    lx: float = 1.0,
    ly: float = 1.0,
) -> list[Path]:
    """Plot model, DNS, and DNS-model passive-tracer fields."""
    error = truth[time_index] - prediction[time_index]
    error_limit = max(float(np.max(np.abs(error))), 1.0e-12)
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    for ax, title, field in zip(
        axes,
        ("Model", "DNS", "Error"),
        (prediction[time_index], truth[time_index], error),
    ):
        if title == "Error":
            image = ax.imshow(
                field.T,
                origin="lower",
                extent=(0, lx, 0, ly),
                cmap="viridis",
                vmin=-error_limit,
                vmax=error_limit,
            )
        else:
            image = ax.imshow(
                field.T,
                origin="lower",
                extent=(0, lx, 0, ly),
                cmap="RdBu_r",
                vmin=-1.0,
                vmax=1.0,
            )
        ax.set_title(title, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
    fig.suptitle(f"$t={time_value:g}$", fontsize=14)
    fig.subplots_adjust(wspace=0.10, top=0.84)
    return _save_figure(fig, output_stem, formats, dpi)
