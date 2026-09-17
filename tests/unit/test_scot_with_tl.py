"""Regression checks for the locked scOT transfer-learning ablation."""

from copy import deepcopy
from pathlib import Path

import pytest

from phase.training.scot_trainer import _validate_transfer_learning_ablation
from phase.utils import load_config


def _config(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    return load_config(root / "configs/ablations/scot_with_tl/re1000.yaml")


def test_scot_with_tl_config_is_locked(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    model = config["model_params"]
    loss = config["loss_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer["param_groups"]
    dataset = config["dataset_params"]
    train = config["train_params"]
    assert model["model_type"] == "poseidon-mhd-finetune"
    assert model["poseidon_model"] == "camlab-ethz/Poseidon-T"
    assert model["load_pretrained_poseidon"] is True
    assert model["out_channels"] == 3
    assert model["poseidon_input_channel_map"] == [1, 2]
    assert model["poseidon_output_channel_map"] == [1, 2]
    assert model["magnetic_channel_index"] == 4
    assert model["use_poseidon_fluid_normalization"] is True
    assert model["magnetic_input_init"] == "mean_velocity"
    assert model["magnetic_output_init"] == "zero"
    assert model["velocity_residual"] is False
    assert model["magnetic_residual"] is True
    assert config["normalization_params"]["type"] == "physics"
    assert config["normalization_params"]["input_norm"] == [1.0, 1.0, 0.0052]
    assert config["normalization_params"]["output_norm"] == [1.0, 1.0, 0.0052]
    assert all(
        config["dataloader_params"][split]["batch_size"] == 16
        for split in ("train", "validation", "test")
    )
    assert dataset["data_path"].endswith("mhd_Re1000_N1000/mhd_data_3channel.npy")
    assert dataset["train_size"] == 800
    assert dataset["val_plus_test_size"] == 200
    assert dataset["seed"] == 42
    assert dataset["sub_t"] == 4
    assert dataset["sub_x"] == 1
    assert dataset["t_range"] == [0.0, 1.0]
    assert loss["nu"] == pytest.approx(1e-3)
    assert loss["eta"] == pytest.approx(1e-3)
    assert loss["rho0"] == pytest.approx(1.0)
    assert loss["magnetic_field_weight"] == pytest.approx(1.0)
    assert loss["log_inactive_derived_components"] is True
    assert loss["data_weight"] == pytest.approx(10.0)
    assert loss["ic_weight"] == pytest.approx(1.0)
    assert loss["pde_weight"] == pytest.approx(1e-3)
    assert loss["constraint_weight"] == pytest.approx(0.1)
    enabled = ("use_data_loss", "use_ic_loss", "use_pde_loss", "use_constraint_loss")
    assert [loss[name] for name in enabled] == [True, True, True, True]
    assert [loss[name] for name in ("u_weight", "v_weight", "A_weight")] == [1.0, 1.0, 5.0]
    assert [loss[name] for name in ("Du_weight", "Dv_weight", "DA_weight")] == [1.0, 1.0, 100.0]
    assert loss["div_vel_weight"] == pytest.approx(1.0)
    assert loss["div_B_weight"] == pytest.approx(0.0)
    assert optimizer["lr"] == pytest.approx(1e-5)
    assert optimizer["weight_decay"] == pytest.approx(1e-2)
    assert optimizer["betas"] == [0.9, 0.999]
    assert groups["pretrained_lr"] == pytest.approx(5e-6)
    assert groups["new_lr"] == pytest.approx(5e-4)
    assert groups["pretrained_weight_decay"] == pytest.approx(1e-2)
    assert groups["new_weight_decay"] == pytest.approx(0.0)
    assert config["train_params"]["epochs"] == 100
    assert config["train_params"]["load_checkpoint"] == ""
    assert optimizer["use_scheduler"] is False
    assert optimizer["scheduler_type"] == "dummy"
    assert train["recipe"] == "scot_with_tl"
    assert train["epochs"] == 100
    assert train["load_checkpoint"] == ""
    assert train["is_finetune"] is True
    assert train["checkpoint_metric"] == "normalized_validation_loss"
    _validate_transfer_learning_ablation(config)


def test_scot_with_tl_guard_rejects_batch_size_drift(monkeypatch, tmp_path):
    config = deepcopy(_config(monkeypatch, tmp_path))
    config["dataloader_params"]["train"]["batch_size"] = 1
    with pytest.raises(ValueError, match="batch_size=16"):
        _validate_transfer_learning_ablation(config)


def test_scot_with_tl_guard_rejects_disabled_pretraining(monkeypatch, tmp_path):
    config = deepcopy(_config(monkeypatch, tmp_path))
    config["model_params"]["load_pretrained_poseidon"] = False
    with pytest.raises(ValueError, match="load_pretrained_poseidon"):
        _validate_transfer_learning_ablation(config)
