"""Regression tests for held-out-test visualization primitives."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phase.visualization import (
    advect_tracer,
    plot_field_comparison,
    plot_pdf_comparison,
    plot_spectrum_comparison,
    plot_tracer_comparison,
    resolve_time_indices,
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


def test_zero_velocity_preserves_postprocessed_tracer_without_diffusion():
    velocity = torch.zeros(4, 16, 16)
    times = np.linspace(0.0, 0.3, 4)
    tracer = advect_tracer(
        velocity, velocity, times, delta=0.05, diffusivity=0.0
    )
    for index in range(1, len(times)):
        assert np.array_equal(tracer[index], tracer[0])


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
