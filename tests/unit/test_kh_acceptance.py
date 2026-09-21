"""Guards for the reduced end-to-end KH acceptance chain."""

from pathlib import Path

import pytest
import yaml

from phase.config_validation import validate_config
from phase.training.dino_trainer import _validate_recipe
from phase.training.scot_trainer import (
    _validate_kh_multi_re,
    _validate_kh_single_re,
)
from phase.utils import load_config


ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "configs" / "acceptance" / "kh_20pct_10ep"


@pytest.mark.parametrize(
    "name",
    (
        "single_re_scot.yaml",
        "single_re_phase.yaml",
        "multi_re_scot.yaml",
        "multi_re_phase.yaml",
    ),
)
def test_kh_acceptance_recipes_preserve_reduced_time_contract(name):
    config = yaml.safe_load((CONFIG_ROOT / name).read_text())
    data = config["dataset_params"]
    train = config["train_params"]

    assert train["epochs"] == 10
    assert train["acceptance_run"] is True
    assert validate_config(config) == []
    if "scot" in name:
        assert data["sub_t"] == 5
        assert data["t_range"] == [0.0, 5.0]
        assert data["train_sample_fraction"] == 0.2
        assert data["validation_sample_fraction"] == 0.1
        assert data["test_sample_fraction"] == 0.1
    else:
        assert data["source_time_range"] == [0.0, 5.0]
        assert data["source_sub_t"] == 5
        assert data["frames_per_trajectory"] == 51
        assert data["source_train_sample_fraction"] == 0.2
        assert data["source_validation_sample_fraction"] == 0.1
        assert data["source_test_sample_fraction"] == 0.1


def test_kh_acceptance_warm_start_chain_is_internal():
    sr_scot = yaml.safe_load((CONFIG_ROOT / "single_re_scot.yaml").read_text())
    sr_phase = yaml.safe_load((CONFIG_ROOT / "single_re_phase.yaml").read_text())
    mr_scot = yaml.safe_load((CONFIG_ROOT / "multi_re_scot.yaml").read_text())
    mr_phase = yaml.safe_load((CONFIG_ROOT / "multi_re_phase.yaml").read_text())

    assert sr_scot["train_params"]["warm_start_checkpoint"] == ""
    assert sr_phase["train_params"]["warm_start_mode"] == "none"
    assert mr_scot["train_params"]["warm_start_checkpoint"].endswith(
        "/kh_20pct_10ep/single_re_scot_re1000.pt"
    )
    assert mr_phase["train_params"]["warm_start_mode"] == "model_weights_only"
    assert mr_phase["train_params"]["warm_start_checkpoint"].endswith(
        "/kh_20pct_10ep/single_re_phase_re1000.pt"
    )


def test_kh_acceptance_diffusion_is_four_field_residual_projection():
    for name in ("single_re_phase.yaml", "multi_re_phase.yaml"):
        config = yaml.safe_load((CONFIG_ROOT / name).read_text())
        model = config["model_params"]
        data = config["dataset_params"]
        assert model["channels"] == 4
        assert model["helmholtz_projection"] is True
        assert model["projection_mode"] == "full_field_residual"
        assert model["re_conditioning"]["enabled"] is False
        assert data["prediction_mode"] == "residual"
        assert data["residual_target"] is True


def test_kh_acceptance_scot_recipes_pass_runtime_guards(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))

    single_re = load_config(CONFIG_ROOT / "single_re_scot.yaml")
    multi_re = load_config(CONFIG_ROOT / "multi_re_scot.yaml")

    _validate_kh_single_re(single_re)
    _validate_kh_multi_re(multi_re)


def test_kh_acceptance_diffusion_recipes_pass_runtime_guards(
    monkeypatch, tmp_path
):
    for name in ("FEATURE_ROOT", "STATS_ROOT", "OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))

    single_re = load_config(CONFIG_ROOT / "single_re_phase.yaml")
    multi_re = load_config(CONFIG_ROOT / "multi_re_phase.yaml")

    assert _validate_recipe(single_re) == "kh_phase_residual_single_re"
    assert _validate_recipe(multi_re) == "kh_phase_residual_multi_re"
