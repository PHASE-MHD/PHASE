"""Focused tests for the first PHASE runtime batch."""

from pathlib import Path

import torch

from phase.activations import create_activation
from phase.optimizers import create_scheduler
from phase.utils import load_config, save_config


def test_activation_factory_registers_standard_activations():
    activation = create_activation("gelu")
    assert isinstance(activation, torch.nn.GELU)


def test_disabled_scheduler_works_through_public_api():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
    scheduler = create_scheduler(
        optimizer, {"optimizer_params": {"use_scheduler": False}}
    )
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)


def test_yaml_config_round_trip(tmp_path: Path):
    path = tmp_path / "config.yaml"
    expected = {"optimizer_params": {"lr": 1.0e-3}}
    save_config(expected, path)
    assert load_config(path) == expected
