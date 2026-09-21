"""Release-level checks for every public experiment recipe."""

from pathlib import Path

import pytest
import yaml

from phase.config_validation import (
    discover_config_paths,
    validate_config,
    validate_config_file,
)


ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize("path", discover_config_paths(ROOT / "configs"))
def test_every_public_config_passes_static_validation(path):
    assert validate_config_file(path, expand_environment=False) == []


def test_residual_recipe_rejects_direct_targets():
    path = ROOT / "configs/turbulence/single_re/phase_re1000.yaml"
    config = yaml.safe_load(path.read_text())
    config["dataset_params"]["prediction_mode"] = "direct"
    errors = validate_config(config)
    assert any("prediction_mode='residual'" in error for error in errors)


def test_conditioner_is_not_treated_as_a_training_recipe():
    path = ROOT / "configs/previous_baseline/dino/conditioner_re1000.yaml"
    assert validate_config_file(path, expand_environment=False) == []
    config = yaml.safe_load(path.read_text())
    config["model_params"]["decoder_layers"] = 1
    assert (
        "previous DINO conditioner architecture is not canonical"
        in validate_config(config)
    )


def test_kh_time_contract_is_checked():
    path = ROOT / "configs/kh/multi_re/phase.yaml"
    config = yaml.safe_load(path.read_text())
    config["dataset_params"]["frames_per_trajectory"] = 41
    errors = validate_config(config)
    assert any("frames_per_trajectory must be 51" in error for error in errors)


def test_check_paths_reports_unexpanded_environment_variables():
    path = ROOT / "configs/previous_baseline/tfno/re1000.yaml"
    config = yaml.safe_load(path.read_text())
    errors = validate_config(config, check_paths=True)
    assert any("unresolved environment variables" in error for error in errors)


def test_malformed_config_reports_errors_instead_of_crashing():
    errors = validate_config({"model_params": {"model_type": "tfno"}})
    assert "missing config.dataset_params" in errors


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            {
                "model_params": [],
                "dataset_params": {},
                "optimizer_params": {},
                "train_params": {},
            },
            "config.model_params must be a mapping",
        ),
        (
            {
                "config_type": "conditioner",
                "model_params": [],
                "dataset_params": {},
                "normalization_params": {},
            },
            "config.model_params must be a mapping",
        ),
        (
            {
                "config_type": "diffusion",
                "model_params": {"model_type": "diffusion"},
                "dataset_params": {},
                "optimizer_params": {},
                "train_params": {},
                "normalization_params": [],
            },
            "config.normalization_params must be a mapping",
        ),
    ],
)
def test_non_mapping_sections_report_errors(config, expected):
    assert expected in validate_config(config)


def test_malformed_sequences_report_errors():
    path = ROOT / "configs/turbulence/multi_re/scot.yaml"
    config = yaml.safe_load(path.read_text())
    config["dataset_params"]["re_values"] = 1000
    config["normalization_params"]["input_norm"] = 1.0
    errors = validate_config(config)
    assert "dataset_params.re_values must be a sequence" in errors
    assert "normalization_params.input_norm must be a sequence" in errors


def test_invalid_yaml_reports_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("model_params: [\n", encoding="utf-8")
    errors = validate_config_file(path)
    assert len(errors) == 1
    assert errors[0].startswith("could not load YAML config")


@pytest.mark.parametrize(
    "key, value, expected",
    [
        (
            "source_output_dt",
            "0.02",
            "dataset_params.source_output_dt must be finite and positive",
        ),
        (
            "source_sub_t",
            "5",
            "dataset_params.source_sub_t must be a positive integer",
        ),
        (
            "frames_per_trajectory",
            51.0,
            "dataset_params.frames_per_trajectory must be a positive integer",
        ),
    ],
)
def test_kh_time_contract_rejects_wrong_types(key, value, expected):
    path = ROOT / "configs/kh/multi_re/phase.yaml"
    config = yaml.safe_load(path.read_text())
    config["dataset_params"][key] = value
    assert expected in validate_config(config)


def test_per_re_path_templates_are_expanded(tmp_path):
    for re_value in (80, 1000):
        (tmp_path / f"Re{re_value}.npz").touch()
    config = {
        "config_type": "conditioner",
        "model_params": {
            "model_type": "tfno",
            "in_channels": 6,
            "out_channels": 3,
            "num_fno_layers": 8,
        },
        "normalization_params": {
            "re_values": [80, 1000],
            "inputs_stats_template": str(tmp_path / "Re{re}.npz"),
        },
        "dataset_params": {
            "data_path": str(tmp_path),
            "train_size": 1,
            "val_plus_test_size": 1,
            "sub_t": 1,
            "sub_x": 1,
        },
    }
    assert validate_config(config, check_paths=True) == []


def test_multi_re_data_files_are_checked(tmp_path):
    data_root = tmp_path / "data"
    output_root = tmp_path / "output"
    output_root.mkdir()
    for re_value in (80, 1000):
        directory = data_root / f"mhd_Re{re_value}_N2"
        directory.mkdir(parents=True)
        (directory / "fields.npy").touch()
    config = {
        "model_params": {"model_type": "poseidon-mhd-re-finetune"},
        "normalization_params": {},
        "dataset_params": {
            "data_root": str(data_root),
            "data_file": "fields.npy",
            "n": 2,
            "re_values": [80, 1000],
            "rem_values": [80, 1000],
            "sub_t": 1,
            "sub_x": 1,
        },
        "loss_params": {
            "nu": 0.001,
            "eta": 0.001,
            "data_weight": 1.0,
            "ic_weight": 1.0,
            "pde_weight": 1.0,
        },
        "dataloader_params": {
            name: {"batch_size": 1} for name in ("train", "validation", "test")
        },
        "optimizer_params": {"optimizer_type": "adamw", "lr": 1e-4},
        "train_params": {
            "epochs": 1,
            "checkpoint_path": str(output_root / "model.pt"),
        },
    }
    assert validate_config(config, check_paths=True) == []
    (data_root / "mhd_Re80_N2" / "fields.npy").unlink()
    errors = validate_config(config, check_paths=True)
    assert any("mhd_Re80_N2/fields.npy" in error for error in errors)
