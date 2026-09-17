"""Shared fixtures for the PHASE unit tests."""

import pytest


@pytest.fixture
def mini_config():
    return {
        "activation_params": {"type": "gelu"},
        "loss_params": {"type": "mse", "reduction": "mean"},
        "optimizer_params": {
            "optimizer_type": "adam",
            "lr": 1.0e-3,
            "betas": [0.9, 0.999],
            "use_scheduler": True,
            "scheduler_type": "multistep",
            "milestones": [10, 20],
            "gamma": 0.5,
        },
    }
