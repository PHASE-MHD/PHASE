from pathlib import Path

import numpy as np
import pytest
import torch

from phase.data import DiffusionDataset
from phase.models.checkpoint_mapping import map_official_tfno_state_dict
from phase.utils import load_config


def _config(monkeypatch, tmp_path):
    for name in ("STATS_ROOT", "FEATURE_ROOT", "OUTPUT_ROOT"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
    path = Path(__file__).parents[2] / "configs/previous_baseline/dino/re1000.yaml"
    return load_config(path)


def test_dino_config_is_full_field_and_starts_from_scratch(monkeypatch, tmp_path):
    config = _config(monkeypatch, tmp_path)
    assert config["dataset_params"]["prediction_mode"] == "direct"
    assert config["dataset_params"]["residual_target"] is False
    assert config["train_params"]["load_checkpoint"] == ""
    assert config["model_params"]["num_sample_steps"] == 32
    assert config["model_params"]["helmholtz_projection"] is False
    assert config["model_params"]["use_vorticity_loss"] is False
    assert config["model_params"]["use_current_loss"] is False
    assert config["train_params"]["epochs"] == 101
    assert config["train_params"]["validation_interval"] == 10
    assert config["train_params"]["checkpoint_metric"] == "denorm_rel_l2"


def test_direct_diffusion_dataset_does_not_replace_target_with_residual(tmp_path):
    inputs = np.ones((2, 3, 2, 4, 4), dtype=np.float32)
    targets = np.full_like(inputs, 3.0)
    path = tmp_path / "features"
    path.mkdir()
    np.save(path / "diff_inputs.npy", inputs)
    np.save(path / "diff_targets.npy", targets)
    config = {
        "model_params": {"channels": 3, "sigma_data": 0.5},
        "dataset_params": {"prediction_mode": "direct", "residual_target": False},
    }
    dataset = DiffusionDataset(path, config=config, split_name="test")
    condition, target = dataset[0]
    assert torch.all(condition == 1.0)
    assert torch.all(target == 3.0)
    assert not torch.allclose(target, target.new_full(target.shape, 2.0))


def test_official_tfno_checkpoint_mapping():
    tensor = torch.tensor([1.0])
    mapped = map_official_tfno_state_dict(
        {
            "decoder_net.device_buffer": tensor,
            "spec_encoder.lift_network.0.conv.weight": tensor,
            "decoder_net.layers.0.linear.weight": tensor,
            "decoder_net.final_layer.linear.bias": tensor,
        }
    )
    assert "decoder_net.device_buffer" not in mapped
    assert "spec_encoder.lift_network.0.linear.weight" in mapped
    assert "decoder_net.net.0.weight" in mapped
    assert "decoder_net.net.2.bias" in mapped


def test_small_edm_forward_and_sampling():
    pytest.importorskip("einops")
    from phase.diffusion import create_diffusion_model

    config = {
        "model_params": {
            "model_type": "elucidated",
            "base_dim": 8,
            "dim_mults": [1, 2],
            "channels": 3,
            "self_condition": True,
            "flash_attn": False,
            "attn_heads": 1,
            "attn_dim_head": 8,
            "image_size": 8,
            "num_sample_steps": 2,
            "sigma_data": 0.5,
            "helmholtz_projection": False,
        }
    }
    model = create_diffusion_model(config)
    target = torch.randn(2, 3, 8, 8)
    condition = torch.randn_like(target)
    loss = model(target, condition)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    sample = model.sample(condition, num_sample_steps=2)
    assert sample.shape == target.shape
    assert torch.isfinite(sample).all()
