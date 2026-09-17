"""Strict compatibility check for the reported Batch 6 checkpoint."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_scot_without_tl_checkpoint_loads_strictly(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    checkpoint_path = os.environ.get("PHASE_SCOT_WITHOUT_TL_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Set PHASE_SCOT_WITHOUT_TL_CHECKPOINT to run this test")

    from phase.models import create_model
    from phase.utils import load_config

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    config = load_config(root / "configs/ablations/scot_without_tl/re1000.yaml")
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["epoch"] == 95
