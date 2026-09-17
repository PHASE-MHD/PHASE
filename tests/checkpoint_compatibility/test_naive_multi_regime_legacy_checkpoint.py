"""Compatibility checks for the Batch 8 warm start and reported checkpoint."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_naive_multi_regime_checkpoints_load(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    warm_path = os.environ.get("PHASE_SCOT_WITH_TL_CHECKPOINT")
    final_path = os.environ.get("PHASE_NAIVE_MULTI_RE_CHECKPOINT")
    if not warm_path or not final_path:
        pytest.skip(
            "Set PHASE_SCOT_WITH_TL_CHECKPOINT and "
            "PHASE_NAIVE_MULTI_RE_CHECKPOINT to run this test"
        )

    from phase.models import create_model
    from phase.optimizers import create_optimizer, create_scheduler
    from phase.training.scot_trainer import _warm_start_model
    from phase.utils import load_config

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    config = load_config(
        root / "configs/ablations/naive_multi_regime/multi_re.yaml"
    )
    model = create_model(config)
    _warm_start_model(model, warm_path)

    projection = model.poseidon.embeddings.patch_embeddings.projection.weight
    assert projection.shape[1] == 7
    assert torch.count_nonzero(projection[:, 5:]).item() == 0

    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameters = model.get_optimizer_parameters(config["optimizer_params"])
    optimizer = create_optimizer(parameters, config)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = create_scheduler(optimizer, config)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    assert checkpoint["epoch"] == 59
    assert checkpoint["loss"] == pytest.approx(5.424514629364014)
    assert sum(parameter.numel() for parameter in model.parameters()) == 20_777_742
    assert [len(group["params"]) for group in optimizer.param_groups] == [843, 1]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-7, 1e-3]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [
        1e-2,
        0.0,
    ]
    assert len(optimizer.state) == 844
    assert scheduler.last_epoch == 60

    re = torch.tensor([80.0, 1000.0, 4500.0])
    expected = (torch.log10(re) - 2.9756) / 0.5417
    assert torch.allclose(model._normalize_re(re), expected)
