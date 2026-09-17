"""Compatibility check for the external legacy tFNO checkpoint."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_legacy_tfno_checkpoint_loads_strictly(monkeypatch, tmp_path):
    pytest.importorskip("tltorch")
    checkpoint_path = os.environ.get("PHASE_LEGACY_TFNO_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Set PHASE_LEGACY_TFNO_CHECKPOINT to run this test")

    from phase.models import create_model
    from phase.utils import load_config

    for name in ("DATA_ROOT", "STATS_ROOT", "OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    root = Path(__file__).parents[2]
    config = load_config(root / "configs/previous_baseline/tfno/re1000.yaml")
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["epoch"] == 99
