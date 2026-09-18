"""Regression tests for the unified held-out-test evaluator."""

from __future__ import annotations

import json

import pytest
import torch

from phase.evaluation.inference import model_family, representation
from phase.evaluation.metrics import (
    EvaluationRecord,
    _spectrum_errors,
    evaluate_records,
)
from phase.evaluation.physics import derive_fields, divergence_2d
from phase.evaluation.reporting import write_evaluation_report


def _periodic_trajectory(nt=3, n=32):
    x = torch.arange(n, dtype=torch.float64) / n
    y = torch.arange(n, dtype=torch.float64) / n
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    potential = torch.sin(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
    stream = torch.sin(2 * torch.pi * xx) * torch.sin(2 * torch.pi * yy)
    ux = 2 * torch.pi * torch.sin(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
    uy = -2 * torch.pi * torch.cos(2 * torch.pi * xx) * torch.sin(2 * torch.pi * yy)
    bx = -2 * torch.pi * torch.sin(2 * torch.pi * xx) * torch.sin(2 * torch.pi * yy)
    by = -2 * torch.pi * torch.cos(2 * torch.pi * xx) * torch.cos(2 * torch.pi * yy)
    direct = torch.stack([ux, uy, bx, by]).unsqueeze(1).repeat(1, nt, 1, 1)
    vecpot = torch.stack([ux, uy, potential]).unsqueeze(1).repeat(1, nt, 1, 1)
    return direct, vecpot, stream


def test_vector_potential_and_direct_b_use_identical_spectral_derivatives():
    direct, vecpot, stream = _periodic_trajectory()
    from_direct = derive_fields(direct, "direct_b")
    from_vecpot = derive_fields(vecpot, "vector_potential")
    for name in ("ux", "uy", "Bx", "By", "omega", "j"):
        assert torch.allclose(from_direct[name], from_vecpot[name], atol=1e-10)
    expected_omega = 8 * torch.pi**2 * stream
    assert torch.allclose(from_direct["omega"][0], expected_omega, atol=1e-10)


def test_divergence_and_metrics_are_zero_for_exact_periodic_prediction():
    direct, _, _ = _periodic_trajectory()
    fields = derive_fields(direct, "direct_b")
    assert torch.sqrt(
        torch.mean(divergence_2d(fields["ux"], fields["uy"]).square())
    ) < 1e-10
    assert torch.sqrt(
        torch.mean(divergence_2d(fields["Bx"], fields["By"]).square())
    ) < 1e-10
    result = evaluate_records(
        [EvaluationRecord(direct, direct, "direct_b", 901, 1000.0)],
        problem="turbulence",
    )
    assert result["aggregate"]
    assert max(abs(value) for value in result["aggregate"].values()) < 1e-10


def test_kh_report_excludes_turbulence_distribution_and_spectrum_metrics():
    direct, _, _ = _periodic_trajectory()
    result = evaluate_records(
        [EvaluationRecord(direct * 2, direct, "direct_b", 902, 1000.0)],
        problem="kh",
    )
    assert result["aggregate"]["rel_l2_ux"] == pytest.approx(1.0)
    assert not any("spectrum" in key for key in result["aggregate"])
    assert not any("pdf" in key for key in result["aggregate"])
    assert not any("kurtosis" in key for key in result["aggregate"])


def test_kh_relative_l2_is_computed_over_the_complete_trajectory():
    truth = torch.zeros(4, 2, 4, 4, dtype=torch.float64)
    truth[:, 0] = 1.0
    truth[:, 1] = 10.0
    prediction = truth.clone()
    prediction[:, 0] = 0.0
    result = evaluate_records(
        [EvaluationRecord(prediction, truth, "direct_b", 905, 1000.0)],
        problem="kh",
    )
    expected = (1.0 / 101.0) ** 0.5
    assert result["aggregate"]["rel_l2_ux"] == pytest.approx(expected)
    assert "complete space-time trajectory" in result["aggregation"]


def test_spectrum_error_preserves_reported_log_ratio_definition():
    truth = torch.ones(16, dtype=torch.float64)
    prediction = truth * 10.0
    low, high = _spectrum_errors(prediction, truth)
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)


def test_duplicate_sample_identity_is_rejected():
    direct, _, _ = _periodic_trajectory()
    record = EvaluationRecord(direct, direct, "direct_b", 903, 80.0)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_records([record, record], problem="kh")


def test_reports_are_test_only_and_record_machine_readable_provenance(tmp_path):
    direct, _, _ = _periodic_trajectory()
    results = evaluate_records(
        [EvaluationRecord(direct, direct, "direct_b", 904, 4500.0)],
        problem="kh",
    )
    metadata = {
        "model": "PHASE",
        "problem": "kh",
        "representation": "direct_b",
        "split": "test",
        "re": 4500.0,
        "config": "/config.yaml",
        "config_sha256": "config-digest",
        "checkpoint": "/checkpoint.pt",
        "checkpoint_sha256": "checkpoint-digest",
        "checkpoint_epoch": 95,
    }
    paths = write_evaluation_report(tmp_path, metadata, results)
    payload = json.loads(paths["json"].read_text())
    assert payload["metadata"]["split"] == "test"
    assert payload["results"]["per_sample"][0]["sample_id"] == 904
    bad = dict(metadata, split="val")
    with pytest.raises(ValueError, match="split='test'"):
        write_evaluation_report(tmp_path / "bad", bad, results)


@pytest.mark.parametrize(
    ("config", "family", "field_representation"),
    [
        ({"model_params": {"model_type": "tfno", "out_channels": 3}}, "tFNO", "vector_potential"),
        (
            {
                "config_type": "diffusion",
                "model_params": {"channels": 3},
                "train_params": {"recipe": "previous_dino"},
            },
            "DINO",
            "vector_potential",
        ),
        (
            {
                "config_type": "diffusion",
                "model_params": {"channels": 4},
                "train_params": {"recipe": "phase_residual_multi_re"},
            },
            "PHASE",
            "direct_b",
        ),
    ],
)
def test_model_family_and_representation_dispatch(config, family, field_representation):
    assert model_family(config) == family
    assert representation(config) == field_representation
