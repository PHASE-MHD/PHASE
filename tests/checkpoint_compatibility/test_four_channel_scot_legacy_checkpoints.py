"""Compatibility checks for the Batch 10 four-channel checkpoints."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_four_channel_checkpoints_load_and_warm_start(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    single_path = os.environ.get("PHASE_SINGLE_RE_FOUR_CHANNEL_CHECKPOINT")
    multi_path = os.environ.get("PHASE_MULTI_RE_FOUR_CHANNEL_CHECKPOINT")
    if not single_path or not multi_path:
        pytest.skip(
            "Set PHASE_SINGLE_RE_FOUR_CHANNEL_CHECKPOINT and "
            "PHASE_MULTI_RE_FOUR_CHANNEL_CHECKPOINT"
        )

    from phase.models import create_model
    from phase.optimizers import create_optimizer
    from phase.training.scot_trainer import _warm_start_model
    from phase.utils import load_config

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    single_config = load_config(
        root / "configs/turbulence/single_re/scot_re1000.yaml"
    )
    multi_config = load_config(
        root / "configs/ablations/four_channel_hp_physics/multi_re.yaml"
    )

    single = create_model(single_config).eval()
    multi = create_model(multi_config).eval()
    single_checkpoint = torch.load(
        single_path, map_location="cpu", weights_only=True
    )
    single.load_state_dict(single_checkpoint["model_state_dict"], strict=True)
    single_optimizer = create_optimizer(
        single.get_optimizer_parameters(single_config["optimizer_params"]),
        single_config,
    )
    single_optimizer.load_state_dict(single_checkpoint["optimizer_state_dict"])
    _warm_start_model(multi, single_path)

    assert single_checkpoint["epoch"] == 98
    assert sum(t.numel() for t in single.state_dict().values()) == 20_778_018
    assert [len(group["params"]) for group in single_optimizer.param_groups] == [839, 5]
    assert [group["lr"] for group in single_optimizer.param_groups] == [5e-6, 5e-4]
    assert [group["weight_decay"] for group in single_optimizer.param_groups] == [
        1e-2, 0.0
    ]
    assert len(multi.expected_warm_start_missing_keys()) == 390

    multi_checkpoint = torch.load(
        multi_path, map_location="cpu", weights_only=True
    )
    multi.load_state_dict(multi_checkpoint["model_state_dict"], strict=True)
    optimizer = create_optimizer(
        multi.get_optimizer_parameters(multi_config["optimizer_params"]),
        multi_config,
    )
    optimizer.load_state_dict(multi_checkpoint["optimizer_state_dict"])

    assert multi_checkpoint["epoch"] == 91
    assert multi_checkpoint["loss"] == pytest.approx(11.978258913993836)
    assert multi_checkpoint["denorm_loss_rel_l2"] == pytest.approx(
        0.03067854103259742
    )
    assert multi_checkpoint["denorm_loss_mse"] == pytest.approx(
        2.7775735347404405e-05
    )
    assert sum(t.numel() for t in multi.state_dict().values()) == 22_291_436
    assert [len(group["params"]) for group in optimizer.param_groups] == [844, 324]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-7, 1e-3]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        1e-2, 0.0
    ]
