"""Regression tests for held-out-test visualization primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt

from phase.visualization import (
    advect_tracer,
    plot_field_comparison,
    plot_pdf_comparison,
    plot_spectrum_comparison,
    plot_tracer_comparison,
    resolve_time_indices,
)
from phase.visualization.plots import (
    _imshow,
    _normalized_spectrum,
    initial_kh_tracer,
)


def _trajectory(nt=3, n=16):
    generator = torch.Generator().manual_seed(14)
    return torch.randn(4, nt, n, n, generator=generator)


def test_time_resolution_uses_physical_nearest_frames():
    times, indices = resolve_time_indices(51, (0.0, 5.0), [0.5, 1.8, 3.5])
    assert indices == [5, 18, 35]
    assert times[indices].tolist() == pytest.approx([0.5, 1.8, 3.5])
    with pytest.raises(ValueError, match="outside"):
        resolve_time_indices(51, (0.0, 5.0), [6.0])
    with pytest.raises(ValueError, match="outside"):
        resolve_time_indices(51, (0.0, 5.0), [-0.01])
    with pytest.raises(ValueError, match="finite"):
        resolve_time_indices(51, (0.0, 5.0), [float("nan")])


def test_image_orientation_and_physical_extent():
    field = torch.arange(12).reshape(3, 4)
    fig, ax = plt.subplots()
    image = _imshow(ax, field, lx=2.0, ly=3.0)
    assert np.array_equal(np.asarray(image.get_array()), field.numpy().T)
    assert tuple(image.get_extent()) == (0.0, 2.0, 0.0, 3.0)
    plt.close(fig)


def test_spectrum_display_normalization_is_per_curve():
    values = _normalized_spectrum(torch.tensor([1.0, 2.0, 3.0]))
    assert float(values.sum()) == pytest.approx(1.0)
    zeros = _normalized_spectrum(torch.zeros(3))
    assert torch.isfinite(zeros).all()
    assert torch.count_nonzero(zeros) == 0


def test_initial_tracer_marks_y_layers_not_x_layers():
    tracer = initial_kh_tracer(8, 16)
    assert np.allclose(tracer, tracer[0:1, :])
    assert not np.allclose(tracer[:, 0:1], tracer)


def test_zero_velocity_preserves_postprocessed_tracer_without_diffusion():
    velocity = torch.zeros(4, 16, 16)
    times = np.linspace(0.0, 0.3, 4)
    tracer = advect_tracer(
        velocity, velocity, times, delta=0.05, diffusivity=0.0
    )
    for index in range(1, len(times)):
        assert np.array_equal(tracer[index], tracer[0])
    with pytest.raises(ValueError, match="nonnegative"):
        advect_tracer(velocity, velocity, times, diffusivity=-0.1)
    with pytest.raises(ValueError, match="finite"):
        advect_tracer(velocity, velocity, times, diffusivity=float("nan"))


def test_all_plot_products_write_nonempty_files(tmp_path):
    truth = _trajectory()
    prediction = truth + 0.01
    common = {
        "prediction": prediction,
        "truth": truth,
        "representation": "direct_b",
        "time_index": 2,
        "time_value": 1.0,
        "formats": ("png",),
        "dpi": 72,
    }
    paths = []
    paths.extend(
        plot_field_comparison(**common, output_stem=tmp_path / "fields")
    )
    paths.extend(
        plot_spectrum_comparison(**common, output_stem=tmp_path / "spectra")
    )
    paths.extend(
        plot_pdf_comparison(**common, output_stem=tmp_path / "pdfs")
    )
    times = np.linspace(0.0, 1.0, 3)
    tracer_true = advect_tracer(truth[0], truth[1], times)
    tracer_pred = advect_tracer(prediction[0], prediction[1], times)
    paths.extend(
        plot_tracer_comparison(
            tracer_pred,
            tracer_true,
            time_index=2,
            time_value=1.0,
            output_stem=tmp_path / "tracer",
            formats=("png",),
            dpi=72,
        )
    )
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
