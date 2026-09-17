"""Strict compatibility check for the reported Batch 7 checkpoint."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_scot_with_tl_checkpoint_loads_strictly(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    checkpoint_path = os.environ.get("PHASE_SCOT_WITH_TL_CHECKPOINT")
    if not checkpoint_path:
        pytest.skip("Set PHASE_SCOT_WITH_TL_CHECKPOINT to run this test")

    from phase.models import create_model
    from phase.optimizers import create_optimizer, create_scheduler
    from phase.utils import load_config

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    config = load_config(root / "configs/ablations/scot_with_tl/re1000.yaml")
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameters = model.get_optimizer_parameters(config["optimizer_params"])
    optimizer = create_optimizer(parameters, config)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = create_scheduler(optimizer, config)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    assert checkpoint["epoch"] == 98
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_776_206
    assert [len(group["params"]) for group in optimizer.param_groups] == [839, 5]
    assert [group["lr"] for group in optimizer.param_groups] == [5e-6, 5e-4]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        1e-2, 0.0
    ]
    assert len(optimizer.state) == 844
    assert scheduler.last_epoch == 99
