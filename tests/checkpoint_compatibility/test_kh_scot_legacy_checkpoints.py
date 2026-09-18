"""Compatibility checks for the canonical t=[0,5] KH scOT checkpoints."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_kh_scot_checkpoints_load_and_warm_start(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    single_path = os.environ.get("PHASE_KH_SINGLE_RE_SCOT_CHECKPOINT")
    multi_path = os.environ.get("PHASE_KH_MULTI_RE_SCOT_CHECKPOINT")
    if not single_path or not multi_path:
        pytest.skip(
            "Set PHASE_KH_SINGLE_RE_SCOT_CHECKPOINT and "
            "PHASE_KH_MULTI_RE_SCOT_CHECKPOINT"
        )

    from phase.models import create_model
    from phase.training.scot_trainer import _warm_start_model
    from phase.utils import load_config

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "kh_data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    single_config = load_config(root / "configs/kh/single_re/scot_re1000.yaml")
    multi_config = load_config(root / "configs/kh/multi_re/scot_t0_5.yaml")

    single = create_model(single_config).eval()
    multi = create_model(multi_config).eval()

    single_checkpoint = torch.load(
        single_path, map_location="cpu", weights_only=True
    )
    single.load_state_dict(single_checkpoint["model_state_dict"], strict=True)
    assert single_checkpoint["epoch"] == 95
    assert sum(t.numel() for t in single.state_dict().values()) == 20_778_018

    _warm_start_model(multi, single_path)
    assert len(multi.expected_warm_start_missing_keys()) == 390

    multi_checkpoint = torch.load(
        multi_path, map_location="cpu", weights_only=True
    )
    multi.load_state_dict(multi_checkpoint["model_state_dict"], strict=True)
    assert multi_checkpoint["epoch"] == 55
    assert multi_checkpoint["denorm_loss_rel_l2"] == pytest.approx(0.025518675602041185)
    assert sum(t.numel() for t in multi.state_dict().values()) == 22_291_436
