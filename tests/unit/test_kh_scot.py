"""Regression checks for the canonical Kelvin-Helmholtz scOT recipes."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from phase.losses.physics_informed import MHDDirectBFieldLoss
from phase.training.scot_trainer import (
    _make_tiny_validation_loader,
    _validate_kh_multi_re,
    _validate_kh_single_re,
)
from phase.utils import load_config


def _configs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "kh_data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    single = load_config(root / "configs/kh/single_re/scot_re1000.yaml")
    multi = load_config(root / "configs/kh/multi_re/scot_t0_5.yaml")
    return single, multi


def test_kh_configs_are_locked(monkeypatch, tmp_path):
    single, multi = _configs(monkeypatch, tmp_path)
    _validate_kh_single_re(single)
    _validate_kh_multi_re(multi)

    assert single["dataset_params"]["t_range"] == [0.0, 5.0]
    assert single["dataset_params"]["sub_t"] == 5
    assert single["normalization_params"]["input_norm"] == [
        1.0, 1.0, 6.69424514e-2, 6.69424514e-2
    ]
    assert multi["normalization_params"]["input_norm"] == [
        1.0, 1.0, 8.06614549e-2, 8.06614549e-2
    ]
    assert multi["dataset_params"]["res_per_batch"] == 10
    assert multi["train_params"]["validation_interval"] == 5
    assert multi["train_params"]["checkpoint_metric"] == "denorm_rel_l2"
    for key in (
        "u_time_loss_mode",
        "v_time_loss_mode",
        "magnetic_time_loss_mode",
        "derived_time_loss_mode",
    ):
        assert multi["loss_params"][key] == "time_relative"


@pytest.mark.parametrize(
    ("which", "path", "value", "message"),
    [
        ("single", ("dataset_params", "t_range"), [0.0, 4.0], "51 frames"),
        (
            "single",
            ("loss_params", "magnetic_time_loss_mode"),
            "global",
            "time-relative",
        ),
        (
            "multi",
            ("normalization_params", "output_norm"),
            [1.0, 1.0, 0.07, 0.07],
            "locked global scale",
        ),
        ("multi", ("dataset_params", "res_per_batch"), 5, "balance all ten"),
        (
            "multi",
            ("train_params", "checkpoint_metric"),
            "normalized_validation_loss",
            "denormalized relative L2",
        ),
        ("multi", ("train_params", "validation_interval"), 1, "full denormalized"),
    ],
)
def test_kh_guards_reject_recipe_drift(
    monkeypatch, tmp_path, which, path, value, message
):
    single, multi = _configs(monkeypatch, tmp_path)
    config = deepcopy(single if which == "single" else multi)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    validator = _validate_kh_single_re if which == "single" else _validate_kh_multi_re
    with pytest.raises(ValueError, match=message):
        validator(config)


def test_time_relative_loss_averages_after_spatial_relative_errors():
    target = torch.tensor([[[[1.0, 1.0]], [[10.0, 10.0]]]])
    prediction = target + 1.0
    loss = MHDDirectBFieldLoss._time_relative_component_loss(
        prediction, target, eps=0.0
    )
    assert loss.item() == pytest.approx(0.55)


def test_time_relative_mode_applies_to_primary_and_derived_fields():
    criterion = MHDDirectBFieldLoss(
        u_time_loss_mode="time_relative",
        v_time_loss_mode="time_relative",
        magnetic_time_loss_mode="time_relative",
        derived_time_loss_mode="time_relative",
        u_time_loss_eps=0.0,
        v_time_loss_eps=0.0,
        magnetic_time_loss_eps=0.0,
        derived_time_loss_eps=0.0,
    )
    target = torch.tensor([[[[1.0, 1.0]], [[10.0, 10.0]]]])
    prediction = target + 1.0
    pred = prediction.unsqueeze(1).repeat(1, 4, 1, 1, 1)
    truth = target.unsqueeze(1).repeat(1, 4, 1, 1, 1)
    _, components = criterion.data_loss(pred, truth, return_components=True)
    assert all(value == pytest.approx(0.55) for value in components.values())

    criterion.curl_2d = lambda qx, qy: qx
    assert criterion.vorticity_loss(prediction, prediction, target, target).item() == pytest.approx(0.55)
    assert criterion.current_loss(prediction, prediction, target, target).item() == pytest.approx(0.55)


class _GroupedDataset(Dataset):
    def __init__(self):
        self.flat_indices_by_re_idx = {
            re_idx: list(range(re_idx * 10, re_idx * 10 + 10))
            for re_idx in range(10)
        }

    def __len__(self):
        return 100

    def __getitem__(self, index):
        return index


def test_tiny_validation_selects_five_samples_from_every_regime():
    val_loader = DataLoader(_GroupedDataset(), batch_size=1)
    tiny = _make_tiny_validation_loader(
        val_loader, samples_per_re=5, batch_size=10, seed=42
    )
    selected = tiny.dataset.indices
    assert len(selected) == 50
    for re_idx in range(10):
        assert sum(re_idx * 10 <= index < (re_idx + 1) * 10 for index in selected) == 5
