"""Compatibility tests for external previous-study DINO artifacts."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_official_conditioner_loads_strictly(monkeypatch, tmp_path):
    pytest.importorskip("tltorch")
    path = os.environ.get("PHASE_OFFICIAL_TFNO_CHECKPOINT")
    if not path:
        pytest.skip("Set PHASE_OFFICIAL_TFNO_CHECKPOINT")
    from phase.models import create_model, map_official_tfno_state_dict
    from phase.utils import load_config

    for name in ("DATA_ROOT", "STATS_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    root = Path(__file__).parents[2]
    config = load_config(root / "configs/previous_baseline/dino/conditioner_re1000.yaml")
    model = create_model(config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(map_official_tfno_state_dict(state), strict=True)


@pytest.mark.checkpoint
def test_epoch100_dino_loads_strictly(monkeypatch, tmp_path):
    pytest.importorskip("einops")
    path = os.environ.get("PHASE_LEGACY_DINO_CHECKPOINT")
    if not path:
        pytest.skip("Set PHASE_LEGACY_DINO_CHECKPOINT")
    from phase.diffusion import create_diffusion_model
    from phase.utils import load_config

    for name in ("STATS_ROOT", "FEATURE_ROOT", "OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    root = Path(__file__).parents[2]
    config = load_config(root / "configs/previous_baseline/dino/re1000.yaml")
    with torch.device("meta"):
        model = create_diffusion_model(config)
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, mmap=True
    )
    model.load_state_dict(
        checkpoint["model_state_dict"], strict=True, assign=True
    )
    assert checkpoint["epoch"] == 100
