"""Integration regression for visualization artifacts and provenance."""

from __future__ import annotations

import json
import runpy
import sys

import torch

from phase.evaluation.metrics import EvaluationRecord


def test_visualization_cli_writes_complete_manifest(monkeypatch, tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("dataset_params: {}\n")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 95}, checkpoint_path)
    output_dir = tmp_path / "output"

    generator = torch.Generator().manual_seed(3)
    truth = torch.randn(4, 3, 8, 8, generator=generator)
    record = EvaluationRecord(
        prediction=truth + 0.01,
        truth=truth,
        representation="direct_b",
        sample_id=977,
        re=1000.0,
    )
    namespace = runpy.run_path("scripts/visualize.py", run_name="visualize_test")
    namespace["main"].__globals__["build_test_inference"] = lambda *args, **kwargs: (
        iter([record]),
        {
            "model": "PHASE",
            "split": "test",
            "checkpoint_epoch": 95,
            "checkpoint_sha256": "mock",
            "sample_id_source": "feature_metadata",
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "visualize.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--problem",
            "turbulence",
            "--re",
            "1000",
            "--sample-id",
            "977",
            "--times",
            "1.0",
            "--products",
            "fields",
            "--formats",
            "png",
            "--dpi",
            "72",
            "--device",
            "cpu",
        ],
    )
    namespace["main"]()

    manifest = json.loads(
        (output_dir / "visualization_manifest.json").read_text()
    )
    assert manifest["split"] == "test"
    assert manifest["sample_id"] == 977
    assert manifest["requested_times"] == [1.0]
    assert manifest["time_values"] == [1.0]
    assert manifest["formats"] == ["png"]
    assert manifest["dpi"] == 72
    assert manifest["products"] == ["fields"]
    assert len(manifest["outputs"]) == 1
    assert len(manifest["outputs"][0]["sha256"]) == 64
    assert (output_dir / manifest["outputs"][0]["path"]).is_file()
