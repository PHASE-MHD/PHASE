"""Small real-CUDA forward/backward acceptance tests for public model families."""

import pytest
import torch


pytestmark = pytest.mark.gpu


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return torch.device("cuda")


def _assert_backward(model, loss):
    assert loss.is_cuda and torch.isfinite(loss)
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def _divergence_rms(qx, qy):
    height, width = qx.shape[-2:]
    kx = 2 * torch.pi * torch.fft.fftfreq(
        height, d=1 / height, device=qx.device, dtype=qx.dtype
    ).reshape(1, height, 1)
    ky = 2 * torch.pi * torch.fft.fftfreq(
        width, d=1 / width, device=qx.device, dtype=qx.dtype
    ).reshape(1, 1, width)
    divergence_hat = 1j * (
        kx * torch.fft.fft2(qx) + ky * torch.fft.fft2(qy)
    )
    if height % 2 == 0:
        divergence_hat[:, height // 2, :] = 0
    if width % 2 == 0:
        divergence_hat[:, :, width // 2] = 0
    return torch.fft.ifft2(divergence_hat).real.square().mean((-2, -1)).sqrt()


def test_tfno_cuda_forward_backward():
    pytest.importorskip("tltorch")
    from phase.models import create_model

    device = _cuda()
    model = create_model(
        {
            "model_params": {
                "model_type": "tfno",
                "model_variant": "3d",
                "in_channels": 6,
                "out_channels": 3,
                "decoder_layers": 1,
                "latent_channels": 4,
                "num_fno_layers": 2,
                "num_fno_modes": 2,
                "padding": [1, 0, 0],
                "padding_type": "constant",
                "activation_fn": "gelu",
                "coord_features": False,
                "rank": 0.5,
                "factorization": "cp",
            }
        }
    ).to(device)
    output = model(torch.randn(1, 6, 5, 8, 8, device=device))
    assert output.shape == (1, 3, 5, 8, 8)
    _assert_backward(model, output.square().mean())


def test_scot_cuda_forward_backward():
    pytest.importorskip("transformers")
    pytest.importorskip("scOT")
    from phase.models import create_model

    device = _cuda()
    model = create_model(
        {
            "model_params": {
                "model_type": "poseidon-mhd-finetune",
                "poseidon_model": None,
                "load_pretrained_poseidon": False,
                "image_size": 64,
                "patch_size": 4,
                "out_channels": 4,
                "magnetic_channel_indices": [4, 5],
                "use_poseidon_fluid_normalization": False,
                "velocity_residual": True,
                "magnetic_residual": True,
                "helmholtz_projection": True,
                "fallback_poseidon_num_channels": 6,
                "fallback_poseidon_num_out_channels": 6,
                "fallback_poseidon_embed_dim": 12,
                "fallback_poseidon_depths": [1, 1, 1, 1],
                "fallback_poseidon_num_heads": [1, 2, 3, 6],
                "fallback_poseidon_skip_connections": [1, 1, 1, 0],
                "window_size": 2,
            }
        }
    ).to(device)
    inputs = torch.randn(1, 7, 2, 64, 64, device=device)
    output = model(inputs)
    assert output.shape == (1, 4, 2, 64, 64)
    _assert_backward(model, output.square().mean())


def _small_diffusion(*, residual):
    from phase.diffusion import create_diffusion_model

    return create_diffusion_model(
        {
            "model_params": {
                "model_type": "elucidated",
                "base_dim": 8,
                "dim_mults": [1, 2],
                "channels": 4 if residual else 3,
                "self_condition": True,
                "flash_attn": False,
                "attn_heads": 1,
                "attn_dim_head": 8,
                "image_size": 8,
                "num_sample_steps": 2,
                "sigma_data": 0.5,
                "helmholtz_projection": residual,
                "project_velocity": True,
                "project_B": True,
                "projection_mode": "full_field_residual" if residual else "output",
            }
        }
    )


def test_dino_full_field_cuda_forward_backward_and_sample():
    pytest.importorskip("einops")
    device = _cuda()
    model = _small_diffusion(residual=False).to(device)
    target = torch.randn(2, 3, 8, 8, device=device)
    condition = torch.randn_like(target)
    loss = model(target, condition)
    _assert_backward(model, loss)
    sample = model.sample(condition, num_sample_steps=2)
    assert sample.shape == target.shape
    assert sample.is_cuda and torch.isfinite(sample).all()


def test_phase_residual_cuda_forward_backward_and_full_field_projection():
    pytest.importorskip("einops")
    from phase.utils.diffusion_tensor_normalization import (
        reconstruct_residual_prediction,
    )

    class IdentityNormalizer:
        def normalize(self, value, re=None):
            return value

        def denormalize(self, value, re=None):
            return value

    device = _cuda()
    model = _small_diffusion(residual=True).to(device)
    condition = torch.randn(2, 4, 8, 8, device=device)
    residual = torch.randn_like(condition)
    normalizer = IdentityNormalizer()
    loss = model(
        residual,
        condition,
        input_normalizer=normalizer,
        target_normalizer=normalizer,
        channel_indices=[0, 1, 2, 3],
    )
    _assert_backward(model, loss)
    predicted_residual = model.sample(
        condition,
        num_sample_steps=2,
    )
    full, _, _ = reconstruct_residual_prediction(
        condition,
        predicted_residual,
        residual,
        input_normalizer=normalizer,
        target_normalizer=normalizer,
        channel_indices=[0, 1, 2, 3],
        projection_module=model.projection_module,
        project_full_field=True,
    )
    for x_channel, y_channel in ((0, 1), (2, 3)):
        rms = _divergence_rms(full[:, x_channel], full[:, y_channel])
        assert rms.max() < 1.0e-4
