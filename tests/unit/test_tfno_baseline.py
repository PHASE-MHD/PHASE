import os
from pathlib import Path

import pytest
import torch

import phase.training.tfno_trainer as tfno_trainer
from phase.losses import create_loss
from phase.physics import compute_constraints
from phase.utils import load_config


def test_tfno_config_is_canonical_and_has_no_warm_start(monkeypatch, tmp_path):
    for name in ("DATA_ROOT", "STATS_ROOT", "OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    path = Path(__file__).parents[2] / "configs/previous_baseline/tfno/re1000.yaml"
    config = load_config(path)
    assert config["model_params"]["num_fno_layers"] == 8
    assert config["model_params"]["latent_channels"] == 32
    assert config["model_params"]["rank"] == 0.5
    assert config["dataset_params"]["train_size"] == 900
    assert config["dataset_params"]["sub_t"] == 4
    assert config["loss_params"]["nu"] == pytest.approx(1e-3)
    assert config["loss_params"]["eta"] == pytest.approx(1e-3)
    assert config["train_params"]["load_checkpoint"] == ""
    assert "$" not in config["dataset_params"]["data_path"]


def test_tfno_trainer_forwards_opt_in_sample_fractions(monkeypatch, tmp_path):
    for name in ("DATA_ROOT", "STATS_ROOT", "OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    path = Path(__file__).parents[2] / (
        "configs/acceptance/ablation_smoke_20pct_10ep/tfno_re1000.yaml"
    )
    config = load_config(path)
    captured = {}

    def stop_after_loader(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("loader captured")

    monkeypatch.setattr(tfno_trainer, "get_dataloaders", stop_after_loader)
    with pytest.raises(RuntimeError, match="loader captured"):
        tfno_trainer.train_tfno(config)

    assert captured["train_sample_fraction"] == pytest.approx(0.2)
    assert captured["validation_sample_fraction"] == pytest.approx(0.1)
    assert captured["test_sample_fraction"] == pytest.approx(0.1)


def test_vector_potential_constraints_are_finite():
    generator = torch.Generator().manual_seed(7)
    fields = torch.randn(2, 5, 8, 8, 3, generator=generator)
    u, v, potential = (fields[..., index] for index in range(3))
    div_u, div_b = compute_constraints(u, v, potential, 1.0, 1.0, 1.0)
    assert div_u.shape == (2, 5, 8, 8)
    assert div_b.shape == (2, 5, 8, 8)
    assert torch.isfinite(div_u).all()
    assert torch.isfinite(div_b).all()
    assert div_b.abs().max() < 1e-4


def test_vector_potential_loss_is_registered_and_differentiable():
    config = {
        "loss_params": {
            "type": "physics-informed",
            "nu": 1e-3,
            "eta": 1e-3,
            "data_weight": 5.0,
            "ic_weight": 1.0,
            "pde_weight": 1.0,
            "constraint_weight": 10.0,
            "DA_weight": 1e6,
            "Lx": 1.0,
            "Ly": 1.0,
            "tend": 1.0,
        }
    }
    criterion = create_loss(config)
    target = torch.randn(1, 3, 5, 8, 8)
    prediction = (target + 0.01 * torch.randn_like(target)).requires_grad_()
    loss = criterion.compute_loss(prediction, target, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_tfno_forward_shape_when_tensorly_torch_is_available():
    pytest.importorskip("tltorch")
    from phase.models import create_model

    model = create_model(
        {
            "model_params": {
                "model_type": "tfno",
                "model_variant": "3d",
                "in_channels": 6,
                "out_channels": 3,
                "decoder_layers": 1,
                "latent_channels": 4,
                "num_fno_layers": 2,
                "num_fno_modes": 2,
                "padding": [1, 0, 0],
                "padding_type": "constant",
                "activation_fn": "gelu",
                "coord_features": False,
                "rank": 0.5,
                "factorization": "cp",
            }
        }
    )
    output = model(torch.randn(1, 6, 5, 8, 8))
    assert output.shape == (1, 3, 5, 8, 8)
