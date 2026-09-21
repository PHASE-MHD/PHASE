"""Compatibility checks for the Batch 9 warm start and reported checkpoint."""

import os
from pathlib import Path

import pytest
import torch


@pytest.mark.checkpoint
def test_gated_adapter_multi_regime_checkpoints_load(monkeypatch, tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    warm_path = os.environ.get("PHASE_SCOT_WITH_TL_CHECKPOINT")
    final_path = os.environ.get("PHASE_GATED_ADAPTER_MULTI_RE_CHECKPOINT")
    if not warm_path or not final_path:
        pytest.skip(
            "Set PHASE_SCOT_WITH_TL_CHECKPOINT and "
            "PHASE_GATED_ADAPTER_MULTI_RE_CHECKPOINT to run this test"
        )

    from phase.models import create_model
    from phase.optimizers import create_optimizer, create_scheduler
    from phase.training.scot_trainer import _warm_start_model
    from phase.utils import load_config

    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    config = load_config(
        root / "configs/ablations/gated_adapter_multi_regime/multi_re.yaml"
    )
    base_config = load_config(
        root / "configs/ablations/scot_with_tl/re1000.yaml"
    )
    base_model = create_model(base_config).eval()
    model = create_model(config).eval()
    assert len(model.deep_re_adapters) == 32
    assert len(model.expected_warm_start_missing_keys()) == 390
    warm_checkpoint = torch.load(warm_path, map_location="cpu", weights_only=False)
    base_model.load_state_dict(warm_checkpoint["model_state_dict"], strict=True)
    _warm_start_model(model, warm_path)

    torch.manual_seed(9)
    inputs = torch.randn(1, 3, 128, 128)
    times = torch.tensor([0.37])
    with torch.no_grad():
        base_prediction = base_model(inputs, times)
        low_re_prediction = model(
            inputs, times, re=torch.tensor([80.0]), rem=torch.tensor([80.0])
        )
        high_re_prediction = model(
            inputs, times, re=torch.tensor([4500.0]), rem=torch.tensor([4500.0])
        )
    assert torch.equal(base_prediction, low_re_prediction)
    assert torch.equal(base_prediction, high_re_prediction)

    assert torch.count_nonzero(model.re_conditioner.net[-1].weight).item() == 0
    assert torch.count_nonzero(model.re_conditioner.net[-1].bias).item() == 0
    for adapter in model.deep_re_adapters:
        assert torch.count_nonzero(adapter.up.weight).item() == 0
        assert torch.count_nonzero(adapter.up.bias).item() == 0
        assert torch.count_nonzero(adapter.gate_net[-1].weight).item() == 0
        assert torch.count_nonzero(adapter.gate_net[-1].bias).item() == 0

    checkpoint = torch.load(final_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    parameters = model.get_optimizer_parameters(config["optimizer_params"])
    optimizer = create_optimizer(parameters, config)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = create_scheduler(optimizer, config)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    assert checkpoint["epoch"] == 90
    assert checkpoint["loss"] == pytest.approx(2.8531549996733667)
    assert checkpoint["denorm_loss_rel_l2"] == pytest.approx(0.024361248968169092)
    assert checkpoint["denorm_loss_mse"] == pytest.approx(2.3291558048470052e-05)
    assert sum(tensor.numel() for tensor in model.state_dict().values()) == 22_289_366
    assert [len(group["params"]) for group in optimizer.param_groups] == [844, 324]
    assert [group["lr"] for group in optimizer.param_groups] == [1e-7, 1e-3]
    assert [group["weight_decay"] for group in optimizer.param_groups] == [1e-2, 0.0]
    assert len(optimizer.state) == 1168
    assert scheduler.last_epoch == 91

    re = torch.tensor([80.0, 1000.0, 4500.0])
    expected = (torch.log10(re) - 2.9756) / 0.5417
    assert torch.allclose(model.re_conditioner._normalize_re(re), expected)
