"""Regression checks for the locked scOT-without-TL ablation."""

from pathlib import Path

import pytest
import torch

from phase.losses import create_loss
from phase.utils import load_config


def _config(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    return load_config(root / "configs/ablations/scot_without_tl/re1000.yaml")


def test_scot_without_tl_config_is_locked(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    model = config["model_params"]
    loss = config["loss_params"]
    assert model["poseidon_model"] is None
    assert model["load_pretrained_poseidon"] is False
    assert model["use_poseidon_fluid_normalization"] is False
    assert model["velocity_residual"] is True
    assert model["magnetic_residual"] is True
    assert config["dataloader_params"]["train"]["batch_size"] == 1
    assert config["dataset_params"]["train_size"] == 800
    assert config["dataset_params"]["val_plus_test_size"] == 200
    assert config["dataset_params"]["sub_t"] == 4
    assert loss["nu"] == pytest.approx(1e-3)
    assert loss["eta"] == pytest.approx(1e-3)
    assert loss["magnetic_field_weight"] == pytest.approx(1.0)
    assert config["optimizer_params"]["lr"] == pytest.approx(5e-4)
    assert config["optimizer_params"]["weight_decay"] == pytest.approx(0.0)
    assert config["train_params"]["load_checkpoint"] == ""


def test_scot_vector_potential_loss_reports_all_components():
    criterion = create_loss(
        {
            "loss_params": {
                "type": "physics-informed",
                "nu": 1e-3,
                "eta": 1e-3,
                "data_weight": 10.0,
                "ic_weight": 1.0,
                "pde_weight": 1e-3,
                "constraint_weight": 0.1,
                "magnetic_field_weight": 1.0,
                "A_weight": 5.0,
                "DA_weight": 100.0,
                "div_B_weight": 0.0,
            }
        }
    )
    target = torch.randn(1, 3, 5, 8, 8)
    prediction = (target + 0.01 * torch.randn_like(target)).requires_grad_()
    loss = criterion.compute_loss(prediction, target, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    for name in ("data_u", "data_v", "data_A", "pde_Du", "pde_Dv", "pde_DA", "magnetic_field_Bx", "magnetic_field_By"):
        assert name in criterion.last_components


def test_scot_forward_shape_when_dependency_is_available():
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    from phase.models import create_model

    model = create_model(
        {
            "model_params": {
                "model_type": "poseidon-mhd-finetune",
                "poseidon_model": None,
                "load_pretrained_poseidon": False,
                "image_size": 64,
                "patch_size": 4,
                "out_channels": 3,
                "use_poseidon_fluid_normalization": False,
                "velocity_residual": True,
                "magnetic_residual": True,
                "fallback_poseidon_num_channels": 5,
                "fallback_poseidon_num_out_channels": 5,
                "fallback_poseidon_embed_dim": 12,
                "fallback_poseidon_depths": [1, 1, 1, 1],
                "fallback_poseidon_num_heads": [1, 2, 3, 6],
                "fallback_poseidon_skip_connections": [1, 1, 1, 0],
                "window_size": 2,
            }
        }
    )
    output = model(torch.randn(1, 6, 3, 64, 64))
    assert output.shape == (1, 3, 3, 64, 64)
