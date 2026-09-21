"""Optional compatibility checks for canonical KH PHASE diffusion checkpoints."""

import gc
import os
from pathlib import Path

import pytest
import torch


def _load_mmap(path):
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


@pytest.mark.checkpoint
def test_kh_single_and_multi_re_diffusion_checkpoints(monkeypatch, tmp_path):
    single_path = os.environ.get("PHASE_KH_SINGLE_RE_DIFFUSION_CHECKPOINT")
    multi_path = os.environ.get("PHASE_KH_MULTI_RE_DIFFUSION_CHECKPOINT")
    if not single_path or not multi_path:
        pytest.skip(
            "Set PHASE_KH_SINGLE_RE_DIFFUSION_CHECKPOINT and "
            "PHASE_KH_MULTI_RE_DIFFUSION_CHECKPOINT"
        )

    for name in ("FEATURE_ROOT", "STATS_ROOT", "OUTPUT_ROOT", "CHECKPOINT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    from phase.diffusion import create_diffusion_model
    from phase.utils import load_config

    root = Path(__file__).parents[2]
    single_config = load_config(
        root / "configs/kh/single_re/phase_re1000.yaml"
    )
    multi_config = load_config(
        root / "configs/kh/multi_re/phase.yaml"
    )
    assert single_config["model_params"] == multi_config["model_params"]

    with torch.device("meta"):
        model = create_diffusion_model(single_config)

    single = _load_mmap(single_path)
    result = model.load_state_dict(
        single["model_state_dict"], strict=True, assign=True
    )
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert len(single["model_state_dict"]) == 349
    assert single["epoch"] == 95
    assert single["denorm_loss_rel_l2"] == pytest.approx(0.01637994165532291)
    del single
    gc.collect()

    multi = _load_mmap(multi_path)
    result = model.load_state_dict(
        multi["model_state_dict"], strict=True, assign=True
    )
    assert not result.missing_keys
    assert not result.unexpected_keys
    assert len(multi["model_state_dict"]) == 349
    assert multi["epoch"] == 95
    assert multi["denorm_loss_rel_l2"] == pytest.approx(0.008567048760131002)
