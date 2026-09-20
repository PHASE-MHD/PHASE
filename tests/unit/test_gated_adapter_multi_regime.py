"""Regression checks for the locked gated-adapter multi-regime ablation."""

from copy import deepcopy
from pathlib import Path

import pytest

from phase.training.scot_trainer import _validate_gated_adapter_multi_re_ablation
from phase.utils import load_config


def _config(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    return load_config(
        root / "configs/ablations/gated_adapter_multi_regime/multi_re.yaml"
    )


def test_gated_adapter_config_is_locked(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    model = config["model_params"]
    conditioning = model["re_conditioning"]
    adapter = conditioning["deep_adapter"]
    dataset = config["dataset_params"]
    loss = config["loss_params"]
    optimizer = config["optimizer_params"]
    groups = optimizer["param_groups"]
    train = config["train_params"]
    regimes = [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]

    assert model["model_type"] == "poseidon-mhd-re-finetune"
    assert model["poseidon_model"] == "camlab-ethz/Poseidon-T"
    assert model["load_pretrained_poseidon"] is True
    assert model["use_poseidon_fluid_normalization"] is True
    assert model["velocity_residual"] is False
    assert model["magnetic_residual"] is True
    assert conditioning["enabled"] is True
    assert conditioning["type"] == "deep_adapter_output_film"
    assert conditioning["hidden_dim"] == 128
    assert conditioning["num_layers"] == 2
    assert conditioning["log_re_mean"] == pytest.approx(2.9756)
    assert conditioning["log_re_std"] == pytest.approx(0.5417)
    assert conditioning["include_rem"] is True
    assert conditioning["reference_re"] == pytest.approx(1000.0)
    assert adapter == {
        "target": "all",
        "bottleneck_dim": 64,
        "hidden_dim": 128,
        "num_layers": 2,
        "adapter_scale": 1.0,
        "gate_type": "channel",
    }
    assert config["normalization_params"]["input_norm"] == [1.0, 1.0, 0.0052]
    assert config["normalization_params"]["output_norm"] == [1.0, 1.0, 0.0052]
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
    assert groups["boundary_group"] == "pretrained"
    assert groups["pretrained_weight_decay"] == pytest.approx(1e-2)
    assert groups["new_weight_decay"] == pytest.approx(0.0)
    assert optimizer["use_scheduler"] is False
    assert train["recipe"] == "gated_adapter_multi_re"
    assert train["epochs"] == 100
    assert train["checkpoint_metric"] == "normalized_validation_loss"
    _validate_gated_adapter_multi_re_ablation(config)


def test_gated_adapter_allows_explicit_short_acceptance_run(
    monkeypatch, tmp_path
):
    config = deepcopy(_config(monkeypatch, tmp_path))
    config["train_params"]["epochs"] = 10
    with pytest.raises(ValueError, match="training length and sample fractions"):
        _validate_gated_adapter_multi_re_ablation(config)

    config["dataset_params"]["train_sample_fraction"] = 0.2
    config["dataset_params"]["validation_sample_fraction"] = 0.1
    config["dataset_params"]["test_sample_fraction"] = 0.1
    config["train_params"]["acceptance_run"] = True
    _validate_gated_adapter_multi_re_ablation(config)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("model_params", "re_conditioning", "type"), "output_film", "conditioning type"),
        (("model_params", "re_conditioning", "deep_adapter", "target"), "encoder", "target all blocks"),
        (("normalization_params", "output_norm"), [1.0, 1.0, 0.01], "output normalization"),
        (("loss_params", "pde_weight"), 1.0, "loss group weights"),
        (("train_params", "checkpoint_metric"), "denorm_rel_l2", "checkpoint selection"),
    ],
)
def test_gated_adapter_guard_rejects_recipe_drift(
    monkeypatch, tmp_path, path, value, message
):
    config = deepcopy(_config(monkeypatch, tmp_path))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        _validate_gated_adapter_multi_re_ablation(config)
