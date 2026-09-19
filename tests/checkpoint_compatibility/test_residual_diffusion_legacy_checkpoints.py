"""Optional compatibility checks for reported DT PHASE diffusion checkpoints."""

import gc
import os
from pathlib import Path

import pytest
import torch


def _load_mmap(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.checkpoint
def test_corrected_single_re_residual_checkpoint(monkeypatch, tmp_path):
    checkpoint_path = os.environ.get("PHASE_SINGLE_RE_DIFFUSION_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Set PHASE_SINGLE_RE_DIFFUSION_CHECKPOINT")

    for name in ("FEATURE_ROOT", "STATS_ROOT", "OUTPUT_ROOT", "CHECKPOINT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    from phase.diffusion import create_diffusion_model
    from phase.utils import load_config

    root = Path(__file__).parents[2]
    config = load_config(root / "configs/turbulence/single_re/phase_re1000.yaml")
    with torch.device("meta"):
        model = create_diffusion_model(config)

    checkpoint = _load_mmap(checkpoint_path)
    result = model.load_state_dict(
        checkpoint["model_state_dict"], strict=True, assign=True
    )
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert len(checkpoint["model_state_dict"]) == 349
    assert checkpoint["epoch"] == 90
    assert checkpoint["denorm_loss_rel_l2"] == pytest.approx(
        0.028242717292159797
    )
    del checkpoint
    gc.collect()


@pytest.mark.checkpoint
def test_historical_single_re_warm_start_and_multi_re_checkpoint(
    monkeypatch, tmp_path
):
    warm_path = os.environ.get("PHASE_HISTORICAL_SR_DIFFUSION_CHECKPOINT")
    final_path = os.environ.get("PHASE_MULTI_RE_DIFFUSION_CHECKPOINT")
    if not warm_path or not final_path:
        pytest.skip(
            "Set PHASE_HISTORICAL_SR_DIFFUSION_CHECKPOINT and "
            "PHASE_MULTI_RE_DIFFUSION_CHECKPOINT"
        )

    for name in ("FEATURE_ROOT", "STATS_ROOT", "OUTPUT_ROOT", "CHECKPOINT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    from phase.diffusion import create_diffusion_model
    from phase.training.dino_trainer import _warm_start_weights
    from phase.utils import load_config

    root = Path(__file__).parents[2]
    config = load_config(root / "configs/turbulence/multi_re/phase.yaml")
    model = create_diffusion_model(config)
    _warm_start_weights(model, warm_path)

    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert checkpoint["epoch"] == 95
    assert checkpoint["denorm_loss_rel_l2"] == pytest.approx(0.01023141382)
