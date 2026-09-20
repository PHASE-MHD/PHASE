from pathlib import Path

import numpy as np
import pytest
import torch

from phase.diffusion.models.helmholtz_projection import HelmholtzProjectionDiffusion
from phase.preprocessing import diffusion_statistics_by_re
from phase.training.dino_trainer import (
    _checkpoint,
    _should_validate,
    _validate_recipe,
    _warm_start_weights,
)
from phase.utils import load_config
from phase.utils.diffusion_tensor_normalization import (
    project_residual_via_full_field,
    reconstruct_residual_prediction,
)


class IdentityNormalizer:
    def normalize(self, value, re=None):
        return value

    def denormalize(self, value, re=None):
        return value


def _config(path, monkeypatch, tmp_path):
    for name in ("FEATURE_ROOT", "STATS_ROOT", "OUTPUT_ROOT", "CHECKPOINT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    return load_config(Path(__file__).parents[2] / path)


def test_single_re_phase_recipe_is_residual_and_random(monkeypatch, tmp_path):
    config = _config("configs/turbulence/single_re/phase_re1000.yaml", monkeypatch, tmp_path)
    assert _validate_recipe(config) == "phase_residual_single_re"
    assert config["model_params"]["projection_mode"] == "full_field_residual"
    assert config["normalization_params"]["type"] == "paired_minmax"
    assert config["train_params"]["warm_start_checkpoint"] == ""
    assert config["train_params"]["validation_interval"] == 10
    assert config["train_params"]["checkpoint_metric"] == "denorm_rel_l2"
    assert config["model_params"]["vorticity_loss_weight"] == 0.0
    assert config["model_params"]["current_loss_weight"] == 0.0
    assert config["optimizer_params"]["param_groups"]["enabled"] is False


def test_multi_re_phase_recipe_preserves_reported_warm_start(monkeypatch, tmp_path):
    config = _config("configs/turbulence/multi_re/phase.yaml", monkeypatch, tmp_path)
    assert _validate_recipe(config) == "phase_residual_multi_re"
    assert config["normalization_params"]["type"] == "per_re_paired_minmax"
    assert config["dataset_params"]["res_per_batch"] == 10
    assert config["train_params"]["warm_start_mode"] == "model_weights_only"
    assert "historical" in config["train_params"]["warm_start_checkpoint"]
    assert config["train_params"]["validation_interval"] == 5
    assert config["train_params"]["checkpoint_metric"] == "denorm_rel_l2"
    assert config["model_params"]["vorticity_loss_weight"] == 0.0
    assert config["model_params"]["current_loss_weight"] == 0.0
    assert config["optimizer_params"]["param_groups"]["enabled"] is False


def test_dt_acceptance_diffusion_configs_form_clean_residual_chain(
    monkeypatch, tmp_path
):
    single = _config(
        "configs/acceptance/dt_20pct_10ep/single_re_phase.yaml",
        monkeypatch,
        tmp_path,
    )
    multi = _config(
        "configs/acceptance/dt_20pct_10ep/multi_re_phase.yaml",
        monkeypatch,
        tmp_path,
    )

    assert _validate_recipe(single) == "phase_residual_single_re"
    assert _validate_recipe(multi) == "phase_residual_multi_re"
    for config in (single, multi):
        assert config["train_params"]["epochs"] == 10
        assert config["train_params"]["acceptance_run"] is True
        assert config["dataset_params"]["prediction_mode"] == "residual"
        assert config["model_params"]["projection_mode"] == "full_field_residual"
        assert [
            config["dataset_params"][key]
            for key in (
                "source_train_sample_fraction",
                "source_validation_sample_fraction",
                "source_test_sample_fraction",
            )
        ] == [0.2, 0.1, 0.1]
    assert single["train_params"]["warm_start_checkpoint"] == ""
    assert multi["train_params"]["warm_start_checkpoint"] == (
        single["train_params"]["checkpoint_path"]
    )


def test_residual_projection_operates_on_reconstructed_full_field():
    projection = HelmholtzProjectionDiffusion(project_velocity=True, project_B=True)
    condition = torch.randn(2, 4, 8, 8)
    residual = torch.randn_like(condition)
    normalizer = IdentityNormalizer()
    projected_residual = project_residual_via_full_field(
        condition,
        residual,
        normalizer,
        normalizer,
        [0, 1, 2, 3],
        projection,
    )
    expected = projection(condition + residual) - condition
    assert torch.allclose(projected_residual, expected, atol=1.0e-6)
    assert not torch.allclose(projected_residual, projection(residual), atol=1.0e-5)


def test_residual_metrics_reconstruct_physical_field():
    condition = torch.full((1, 4, 4, 4), 3.0)
    pred_residual = torch.full_like(condition, 0.5)
    true_residual = torch.full_like(condition, 1.0)
    normalizer = IdentityNormalizer()
    pred, truth, recovered_condition = reconstruct_residual_prediction(
        condition,
        pred_residual,
        true_residual,
        input_normalizer=normalizer,
        target_normalizer=normalizer,
        channel_indices=[0, 1, 2, 3],
    )
    assert torch.all(pred == 3.5)
    assert torch.all(truth == 4.0)
    assert torch.equal(recovered_condition, condition)


def test_per_re_statistics_use_residual_targets(tmp_path):
    path = tmp_path / "train"
    path.mkdir()
    inputs = np.ones((4, 4, 2, 2, 2), dtype=np.float32)
    targets = inputs.copy()
    targets[:2] += 2.0
    targets[2:] += 5.0
    np.save(path / "diff_inputs.npy", inputs)
    np.save(path / "diff_targets.npy", targets)
    np.save(path / "re.npy", np.array([80, 80, 1000, 1000], dtype=np.float32))
    stats = diffusion_statistics_by_re(path, prediction_mode="residual", chunk_size=1)
    assert np.allclose(stats[80.0]["targets"]["mean"], 2.0)
    assert np.allclose(stats[1000.0]["targets"]["mean"], 5.0)


def test_residual_validation_schedule_matches_reported_runs():
    sr_epochs = [
        epoch
        for epoch in range(100)
        if _should_validate(epoch, 100, 10, "phase_residual_single_re")
    ]
    mr_epochs = [
        epoch
        for epoch in range(100)
        if _should_validate(epoch, 100, 5, "phase_residual_multi_re")
    ]
    assert sr_epochs == list(range(10, 100, 10))
    assert mr_epochs == list(range(5, 100, 5))
    assert _should_validate(0, 100, 10, "previous_dino")
    assert _should_validate(99, 100, 10, "previous_dino")
    with pytest.raises(ValueError, match="validation_interval"):
        _should_validate(1, 100, 0, "phase_residual_single_re")


def test_checkpoint_serializes_selected_denorm_metric(tmp_path):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    path = tmp_path / "checkpoint.pt"
    metrics = (0.25, 0.125, 0.0625)
    _checkpoint(
        path, model, optimizer, scheduler, 5, metrics, selected_metric_index=1
    )
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["loss"] == metrics[1]
    assert saved["model_val_loss"] == metrics[0]
    assert saved["denorm_loss_rel_l2"] == metrics[1]
    assert saved["denorm_loss_mse"] == metrics[2]

    baseline_path = tmp_path / "baseline_checkpoint.pt"
    _checkpoint(baseline_path, model, optimizer, scheduler, 5, metrics)
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    assert baseline["loss"] == metrics[0]


def test_weights_only_warm_start_is_strict_and_does_not_restore_epoch(tmp_path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 90, "model_state_dict": source.state_dict()}, path)
    _warm_start_weights(target, path)
    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)

    incompatible = torch.nn.Linear(4, 2)
    with pytest.raises(RuntimeError):
        _warm_start_weights(incompatible, path)
