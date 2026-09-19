"""Synthetic integration tests for all evaluation inference adapters."""

from __future__ import annotations

import torch
from torch import nn

import phase.evaluation.inference as inference
from phase.evaluation.metrics import evaluate_records


class _IdentityNormalizer:
    def normalize(self, value, re=None, metadata=None):
        return value

    def denormalize(self, value, re=None, metadata=None):
        return value

    def to(self, device):
        return self


class _FakeModel(nn.Module):
    def __init__(self, channels, residual=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.channels = channels
        self.projection_mode = "full_field_residual" if residual else ""
        self.projection_module = nn.Identity()

    def forward(self, value):
        return value

    def sample(self, condition, num_sample_steps, re=None, rem=None):
        if self.projection_mode:
            return torch.zeros_like(condition)
        return condition


def test_deterministic_tfno_and_scot_adapters(monkeypatch, tmp_path):
    class Dataset:
        indices = [977]
        normalizer = _IdentityNormalizer()

    class Loader:
        dataset = Dataset()

        def __iter__(self):
            value = torch.randn(1, 4, 3, 16, 16)
            yield value, value.clone()

    monkeypatch.setattr(
        inference, "_deterministic_test_loader", lambda config, num_workers: Loader()
    )
    monkeypatch.setattr(inference, "create_model", lambda config: _FakeModel(4))

    checkpoint = tmp_path / "deterministic.pt"
    torch.save(
        {"model_state_dict": _FakeModel(4).state_dict(), "epoch": 7},
        checkpoint,
    )
    for model_type, family in (
        ("tfno", "tFNO"),
        ("poseidon-mhd-finetune", "scOT"),
    ):
        config = {
            "model_params": {"model_type": model_type, "out_channels": 4},
            "dataset_params": {
                "data_path": "/unused",
                "train_size": 1,
                "val_plus_test_size": 1,
            },
            "normalization_params": {"type": "identity"},
        }
        records, details = inference.build_test_inference(
            config,
            checkpoint,
            target_re=1000,
            device=torch.device("cpu"),
            max_samples=1,
        )
        results = evaluate_records(records, problem="kh")
        assert details["model"] == family
        assert details["checkpoint_epoch"] == 7
        assert results["number_of_samples"] == 1
        assert results["aggregate"]["rel_l2_ux"] == 0.0


def test_deterministic_adapter_selects_exact_sample_id(monkeypatch, tmp_path):
    class Dataset:
        indices = [101, 977]
        normalizer = _IdentityNormalizer()

    class Loader:
        dataset = Dataset()

        def __iter__(self):
            for value in (1.0, 2.0):
                trajectory = torch.full((1, 4, 3, 8, 8), value)
                yield trajectory, trajectory.clone()

    monkeypatch.setattr(
        inference, "_deterministic_test_loader", lambda config, num_workers: Loader()
    )
    monkeypatch.setattr(inference, "create_model", lambda config: _FakeModel(4))
    checkpoint = tmp_path / "deterministic.pt"
    torch.save(
        {"model_state_dict": _FakeModel(4).state_dict(), "epoch": 7}, checkpoint
    )
    config = {
        "model_params": {"model_type": "poseidon-mhd-finetune", "out_channels": 4},
        "dataset_params": {
            "data_path": "/unused",
            "train_size": 1,
            "val_plus_test_size": 2,
        },
        "normalization_params": {"type": "identity"},
    }
    records, details = inference.build_test_inference(
        config,
        checkpoint,
        target_re=1000,
        device=torch.device("cpu"),
        max_samples=1,
        sample_ids={977},
    )
    records = list(records)
    assert [record.sample_id for record in records] == [977]
    assert torch.all(records[0].prediction == 2.0)
    assert details["requested_sample_ids"] == [977]


def test_full_field_dino_and_residual_phase_adapters(monkeypatch, tmp_path):
    for residual in (False, True):
        channels = 4 if residual else 3
        condition = torch.randn(3, channels, 16, 16)
        target = torch.randn_like(condition) * 0.05 if residual else condition.clone()
        metadata = {
            "re": torch.full((3,), 1000.0),
            "rem": torch.full((3,), 1000.0),
            "sample_id": torch.full((3,), 977, dtype=torch.long),
        }

        class Dataset:
            residual_target = residual
            channel_indices = None
            input_normalizer = _IdentityNormalizer()
            target_normalizer = _IdentityNormalizer()
            sample_id_per_sample = torch.tensor([977])

        class Loader:
            dataset = Dataset()

            def __iter__(self):
                yield condition, target, metadata

        monkeypatch.setattr(
            inference,
            "get_diffusion_test_dataloader",
            lambda config, num_workers=0: Loader(),
        )
        monkeypatch.setattr(
            inference,
            "create_diffusion_model",
            lambda config: _FakeModel(channels, residual),
        )
        checkpoint = tmp_path / f"diffusion_{residual}.pt"
        torch.save(
            {
                "model_state_dict": _FakeModel(channels, residual).state_dict(),
                "epoch": 12,
            },
            checkpoint,
        )
        config = {
            "config_type": "diffusion",
            "model_params": {"channels": channels},
            "train_params": {
                "recipe": (
                    "phase_residual_multi_re" if residual else "previous_dino"
                )
            },
        }
        records, details = inference.build_test_inference(
            config,
            checkpoint,
            target_re=1000,
            device=torch.device("cpu"),
            max_samples=1,
            num_sample_steps=2,
        )
        records = list(records)
        results = evaluate_records(records, problem="kh")
        assert records[0].sample_id == 977
        assert records[0].prediction.shape == (channels, 3, 16, 16)
        assert details["checkpoint_epoch"] == 12
        if residual:
            assert details["model"] == "PHASE"
            assert results["aggregate"]["rel_l2_ux"] > 0.0
        else:
            assert details["model"] == "DINO"
            assert results["aggregate"]["rel_l2_ux"] == 0.0
