from pathlib import Path

import pytest
import torch

from phase.training.dino_trainer import (
    _resume_training_state,
    _should_validate,
    _validate_recipe,
)
from phase.utils import load_config


def _config(path, monkeypatch, tmp_path):
    for name in ("FEATURE_ROOT", "STATS_ROOT", "OUTPUT_ROOT", "CHECKPOINT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return load_config(Path(__file__).parents[2] / path)


def test_kh_single_re_diffusion_is_four_field_residual(monkeypatch, tmp_path):
    config = _config(
        "configs/kh/single_re/residual_diffusion_re1000.yaml",
        monkeypatch,
        tmp_path,
    )
    assert _validate_recipe(config) == "kh_phase_residual_single_re"
    assert config["dataset_params"]["source_time_range"] == [0.0, 5.0]
    assert config["dataset_params"]["source_sub_t"] == 5
    assert config["dataset_params"]["frames_per_trajectory"] == 51
    assert config["normalization_params"]["type"] == "paired_minmax"
    assert config["model_params"]["channels"] == 4
    assert config["model_params"]["projection_mode"] == "full_field_residual"
    assert config["model_params"]["num_sample_steps"] == 32
    assert config["train_params"]["warm_start_checkpoint"] == ""
    assert config["train_params"]["validation_interval"] == 5


def test_kh_multi_re_diffusion_uses_single_re_weights_only(monkeypatch, tmp_path):
    config = _config(
        "configs/kh/multi_re/residual_diffusion_t0_5.yaml",
        monkeypatch,
        tmp_path,
    )
    assert _validate_recipe(config) == "kh_phase_residual_multi_re"
    assert config["normalization_params"]["type"] == "per_re_paired_minmax"
    assert len(config["normalization_params"]["re_values"]) == 10
    assert config["dataset_params"]["balanced_re_batches"] is True
    assert config["dataset_params"]["res_per_batch"] == 10
    assert config["dataloader_params"]["train"]["num_workers"] == 8
    assert config["train_params"]["warm_start_mode"] == "model_weights_only"
    assert config["train_params"]["warm_start_checkpoint"].endswith(
        "kh_single_re_phase_re1000.pt"
    )
    assert config["model_params"]["re_conditioning"]["enabled"] is False


def test_kh_diffusion_architecture_matches_between_stages(monkeypatch, tmp_path):
    single = _config(
        "configs/kh/single_re/residual_diffusion_re1000.yaml",
        monkeypatch,
        tmp_path,
    )
    multi = _config(
        "configs/kh/multi_re/residual_diffusion_t0_5.yaml",
        monkeypatch,
        tmp_path,
    )
    assert single["model_params"] == multi["model_params"]
    assert single["optimizer_params"] == multi["optimizer_params"]


def test_kh_residual_validation_runs_every_fifth_epoch():
    for recipe in (
        "kh_phase_residual_single_re",
        "kh_phase_residual_multi_re",
    ):
        epochs = [
            epoch
            for epoch in range(100)
            if _should_validate(epoch, 100, 5, recipe)
        ]
        assert epochs == list(range(5, 100, 5))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("source_time_range", [0.0, 4.0], r"t=\[0,5\]"),
        ("source_sub_t", 4, "51 KH frames"),
        ("frames_per_trajectory", 41, "51 KH frames"),
    ],
)
def test_kh_recipe_rejects_wrong_time_contract(
    monkeypatch, tmp_path, key, value, message
):
    config = _config(
        "configs/kh/single_re/residual_diffusion_re1000.yaml",
        monkeypatch,
        tmp_path,
    )
    config["dataset_params"][key] = value
    with pytest.raises(ValueError, match=message):
        _validate_recipe(config)


def test_kh_recipe_rejects_diffusion_re_conditioning(monkeypatch, tmp_path):
    config = _config(
        "configs/kh/multi_re/residual_diffusion_t0_5.yaml",
        monkeypatch,
        tmp_path,
    )
    config["model_params"]["re_conditioning"]["enabled"] = True
    with pytest.raises(ValueError, match="no diffusion Re conditioning"):
        _validate_recipe(config)


def test_full_state_resume_restores_epoch_optimizer_and_metric(tmp_path):
    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(source.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100
    )
    loss = source(torch.ones(1, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    path = tmp_path / "resume.pt"
    torch.save(
        {
            "epoch": 40,
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "denorm_loss_rel_l2": 0.125,
        },
        path,
    )

    target = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.AdamW(
        target.parameters(), lr=1.0e-3
    )
    target_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        target_optimizer, T_max=100
    )
    start_epoch, best = _resume_training_state(
        target, target_optimizer, target_scheduler, path
    )
    assert start_epoch == 41
    assert best == pytest.approx(0.125)
    assert target_scheduler.last_epoch == scheduler.last_epoch
    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)
