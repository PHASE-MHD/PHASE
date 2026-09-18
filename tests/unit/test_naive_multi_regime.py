"""Regression checks for the locked naive multi-regime scOT ablation."""

from copy import deepcopy
from pathlib import Path

import pytest

from phase.data import BalancedReBatchSampler
from phase.training.scot_trainer import _validate_naive_multi_re_ablation
from phase.utils import load_config


def _config(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    return load_config(
        root / "configs/ablations/naive_multi_regime/multi_re.yaml"
    )


def test_naive_multi_regime_config_is_locked(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    model = config["model_params"]
    conditioning = model["re_input_conditioning"]
    dataset = config["dataset_params"]
    loss = config["loss_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer["param_groups"]
    train = config["train_params"]
    regimes = [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]

    assert model["model_type"] == "poseidon-mhd-re-input-finetune"
    assert model["poseidon_model"] == "camlab-ethz/Poseidon-T"
    assert model["load_pretrained_poseidon"] is True
    assert model["use_poseidon_fluid_normalization"] is True
    assert model["velocity_residual"] is False
    assert model["magnetic_residual"] is True
    assert conditioning == {
        "log_re_mean": 2.9756,
        "log_re_std": 0.5417,
        "include_rem": True,
        "reference_re": 1000.0,
        "channel_init": "zero",
    }
    assert config["normalization_params"]["input_norm"] == [1.0, 1.0, 0.0052]
    assert config["normalization_params"]["output_norm"] == [1.0, 1.0, 0.0052]
    assert dataset["dataset_type"] == "multi_re"
    assert dataset["re_values"] == regimes
    assert dataset["rem_values"] == regimes
    assert dataset["train_size_per_re"] == 800
    assert dataset["val_plus_test_size_per_re"] == 200
    assert dataset["seed"] == 42
    assert dataset["sub_t"] == 4
    assert dataset["sub_x"] == 1
    assert dataset["balanced_re_batches"] is True
    assert dataset["res_per_batch"] == 10
    assert all(
        config["dataloader_params"][split]["batch_size"] == 1
        for split in ("train", "validation", "test")
    )
    assert loss["nu"] == pytest.approx(1e-3)
    assert loss["eta"] == pytest.approx(1e-3)
    assert [loss[name] for name in ("data_weight", "ic_weight", "pde_weight", "constraint_weight")] == [10.0, 1.0, 0.001, 0.1]
    assert [loss[name] for name in ("u_weight", "v_weight", "A_weight")] == [1.0, 1.0, 5.0]
    assert [loss[name] for name in ("Du_weight", "Dv_weight", "DA_weight")] == [1.0, 1.0, 100.0]
    assert groups["pretrained_lr"] == pytest.approx(1e-7)
    assert groups["new_lr"] == pytest.approx(1e-3)
    assert groups["pretrained_weight_decay"] == pytest.approx(1e-2)
    assert groups["new_weight_decay"] == pytest.approx(0.0)
    assert optimizer["use_scheduler"] is False
    assert train["recipe"] == "naive_multi_re"
    assert train["epochs"] == 100
    assert train["warm_start_checkpoint"].endswith(
        "checkpoints/scot_with_tl_re1000.pt"
    )
    assert train["load_checkpoint"] == ""
    assert train["checkpoint_metric"] == "normalized_validation_loss"
    _validate_naive_multi_re_ablation(config)


def test_naive_multi_regime_guard_rejects_regime_drift(monkeypatch, tmp_path):
    config = deepcopy(_config(monkeypatch, tmp_path))
    config["dataset_params"]["re_values"][-1] = 5000
    with pytest.raises(ValueError, match="locked ten Re values"):
        _validate_naive_multi_re_ablation(config)


def test_naive_multi_regime_guard_rejects_scientific_recipe_drift(
    monkeypatch, tmp_path
):
    config = deepcopy(_config(monkeypatch, tmp_path))
    config["normalization_params"]["output_norm"][-1] = 0.01
    with pytest.raises(ValueError, match="output normalization"):
        _validate_naive_multi_re_ablation(config)

    config = deepcopy(_config(monkeypatch, tmp_path))
    config["loss_params"]["pde_weight"] = 1.0
    with pytest.raises(ValueError, match="loss group weights"):
        _validate_naive_multi_re_ablation(config)

    config = deepcopy(_config(monkeypatch, tmp_path))
    config["train_params"]["checkpoint_metric"] = "denorm_rel_l2"
    with pytest.raises(ValueError, match="checkpoint selection"):
        _validate_naive_multi_re_ablation(config)


def test_naive_multi_regime_guard_requires_weight_only_warm_start(
    monkeypatch, tmp_path
):
    config = deepcopy(_config(monkeypatch, tmp_path))
    config["train_params"]["warm_start_checkpoint"] = ""
    with pytest.raises(ValueError, match="warm_start_checkpoint"):
        _validate_naive_multi_re_ablation(config)


def test_balanced_sampler_emits_one_sample_per_regime():
    dataset = type("Dataset", (), {})()
    dataset.flat_indices_by_re_idx = {
        regime: list(range(regime * 4, regime * 4 + 4)) for regime in range(10)
    }
    sampler = BalancedReBatchSampler(
        dataset, res_per_batch=10, batches_per_epoch=3, shuffle=True, seed=42
    )
    batches = list(iter(sampler))
    assert len(batches) == 3
    for batch in batches:
        assert len(batch) == 10
        assert sorted(index // 4 for index in batch) == list(range(10))
