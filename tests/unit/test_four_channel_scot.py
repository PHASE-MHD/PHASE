"""Regression checks for the locked four-channel scOT recipes."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from phase.losses.physics_informed import MHDDirectBFieldLoss
from phase.models.scot_mhd import PoseidonMHDFinetune
from phase.training.scot_trainer import (
    _validate_four_channel_multi_re,
    _validate_four_channel_single_re,
)
from phase.utils import load_config


def _configs(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path / "output"))
    root = Path(__file__).parents[2]
    single = load_config(root / "configs/turbulence/single_re/scot_re1000.yaml")
    multi = load_config(
        root / "configs/ablations/four_channel_hp_physics/multi_re.yaml"
    )
    return single, multi


def test_four_channel_configs_are_locked(monkeypatch, tmp_path):
    single, multi = _configs(monkeypatch, tmp_path)
    _validate_four_channel_single_re(single)
    _validate_four_channel_multi_re(multi)

    common = single["model_params"]
    assert common["out_channels"] == 4
    assert common["magnetic_channel_indices"] == [4, 5]
    assert common["magnetic_residual"] is True
    assert common["helmholtz_projection"] is True
    assert common["project_velocity"] is True
    assert common["project_magnetic"] is True
    assert single["normalization_params"]["input_norm"] == [
        1.0, 1.0, 4.27121774e-3, 4.27121774e-3
    ]
    assert single["dataloader_params"]["train"]["batch_size"] == 16
    assert multi["dataloader_params"]["train"]["batch_size"] == 1
    assert multi["dataset_params"]["res_per_batch"] == 10
    assert multi["optimizer_params"]["param_groups"]["boundary_group"] == "pretrained"


@pytest.mark.parametrize(
    ("which", "path", "value", "message"),
    [
        ("single", ("model_params", "out_channels"), 3, "four physical fields"),
        ("single", ("loss_params", "current_weight"), 1.0, "vorticity/current"),
        ("single", ("normalization_params", "output_norm"), [1, 1, .01, .01], "locked global scale"),
        ("multi", ("dataset_params", "res_per_batch"), 5, "balanced ten-regime"),
        ("multi", ("optimizer_params", "param_groups", "boundary_group"), "new", "pretrained group"),
        ("multi", ("train_params", "warm_start_checkpoint"), "", "warm-start"),
    ],
)
def test_four_channel_guards_reject_recipe_drift(
    monkeypatch, tmp_path, which, path, value, message
):
    single, multi = _configs(monkeypatch, tmp_path)
    config = deepcopy(single if which == "single" else multi)
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    validator = (
        _validate_four_channel_single_re
        if which == "single"
        else _validate_four_channel_multi_re
    )
    with pytest.raises(ValueError, match=message):
        validator(config)


def _projection_shell():
    model = PoseidonMHDFinetune.__new__(PoseidonMHDFinetune)
    torch.nn.Module.__init__(model)
    model.helmholtz_projection = True
    model.project_velocity = True
    model.project_magnetic = True
    model.out_channels = 4
    model.helmholtz_domain_size_x = 1.0
    model.helmholtz_domain_size_y = 1.0
    return model


def _divergence_rms(qx, qy):
    height, width = qx.shape[-2:]
    kx = 2 * torch.pi * torch.fft.fftfreq(
        height, d=1 / height, dtype=qx.dtype
    ).reshape(1, height, 1)
    ky = 2 * torch.pi * torch.fft.fftfreq(
        width, d=1 / width, dtype=qx.dtype
    ).reshape(1, 1, width)
    div_hat = 1j * (
        kx * torch.fft.fft2(qx) + ky * torch.fft.fft2(qy)
    )
    if height % 2 == 0:
        div_hat[:, height // 2, :] = 0
    if width % 2 == 0:
        div_hat[:, :, width // 2] = 0
    return torch.fft.ifft2(div_hat).real.square().mean().sqrt()


def test_helmholtz_projection_and_paired_normalization_commute():
    torch.manual_seed(17)
    model = _projection_shell()
    fields = torch.randn(2, 4, 32, 32, dtype=torch.float64)
    projected = model.project_fields_helmholtz(fields)

    assert _divergence_rms(projected[:, 0], projected[:, 1]) < 1e-12
    assert _divergence_rms(projected[:, 2], projected[:, 3]) < 1e-12
    assert torch.allclose(
        projected.mean((-2, -1)), fields.mean((-2, -1)), atol=1e-14, rtol=0
    )

    scales = fields.new_tensor([1.0, 1.0, 4.27121774e-3, 4.27121774e-3])
    scales = scales.view(1, 4, 1, 1)
    normalized_path = model.project_fields_helmholtz(fields / scales) * scales
    assert torch.allclose(normalized_path, projected, atol=1e-13, rtol=0)


def test_direct_b_loss_uses_per_sample_transport_coefficients():
    criterion = MHDDirectBFieldLoss(
        nu=1e-3,
        eta=1e-3,
        data_weight=10,
        ic_weight=1,
        pde_weight=1e-3,
        constraint_weight=0,
        vorticity_weight=2,
        current_weight=5,
        use_pde_loss=True,
        Bx_weight=5,
        By_weight=5,
        DBx_weight=100,
        DBy_weight=100,
        div_vel_weight=0,
        div_B_weight=0,
    )
    pred = torch.zeros(2, 4, 3, 8, 8)
    metadata = {
        "re": torch.tensor([80.0, 4500.0]),
        "rem": torch.tensor([80.0, 4500.0]),
    }
    nu, eta = criterion._transport_coefficients(metadata, pred)
    expected = torch.tensor([1 / 80, 1 / 4500]).view(2, 1, 1, 1)
    assert torch.allclose(nu, expected)
    assert torch.allclose(eta, expected)
