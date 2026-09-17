"""Tests for the optimizer modules."""

import pytest
import torch
import torch.nn as nn
from phase.optimizers.optimizer_factory import create_optimizer
from phase.optimizers.standard_optimizers import (
    create_adam_optimizer,
    create_adamw_optimizer,
    create_sgd_optimizer,
)


def test_optimizer_factory(mini_config):
    """Test optimizer factory creates the proper optimizer."""
    # Create a simple model for testing
    model = nn.Linear(10, 1)

    # Update config to specify optimizer type
    config = mini_config.copy()
    config["optimizer_params"] = {
        "optimizer_type": "adam",
        "lr": 0.001,
        "betas": [0.9, 0.999],
    }

    # Create optimizer from config
    optimizer = create_optimizer(model.parameters(), config)

    # Check type
    assert isinstance(
        optimizer, torch.optim.Adam
    ), "Factory should create Adam optimizer for 'adam' type"

    # Test with different optimizer types
    adamw_config = {"optimizer_params": {"optimizer_type": "adamw", "lr": 0.001}}
    adamw_optimizer = create_optimizer(model.parameters(), adamw_config)
    assert isinstance(
        adamw_optimizer, torch.optim.AdamW
    ), "Factory should create AdamW optimizer for 'adamw' type"

    sgd_config = {
        "optimizer_params": {"optimizer_type": "sgd", "lr": 0.01, "momentum": 0.9}
    }
    sgd_optimizer = create_optimizer(model.parameters(), sgd_config)
    assert isinstance(
        sgd_optimizer, torch.optim.SGD
    ), "Factory should create SGD optimizer for 'sgd' type"


def test_optimizer_parameters():
    """Test optimizer parameters are properly set."""
    # Create a simple model
    model = nn.Linear(10, 1)

    # Test Adam optimizer settings
    adam_params = {
        "lr": 0.002,
        "betas": (0.8, 0.888),
        "eps": 1e-6,
        "weight_decay": 0.01,
    }
    adam_opt = create_adam_optimizer(model.parameters(), adam_params)

    assert adam_opt.param_groups[0]["lr"] == 0.002, "Learning rate not set correctly"
    assert adam_opt.param_groups[0]["betas"] == (0.8, 0.888), "Betas not set correctly"
    assert adam_opt.param_groups[0]["eps"] == 1e-6, "Epsilon not set correctly"
    assert (
        adam_opt.param_groups[0]["weight_decay"] == 0.01
    ), "Weight decay not set correctly"

    # Test AdamW optimizer settings
    adamw_params = {
        "lr": 0.003,
        "weight_decay": 0.1,
    }
    adamw_opt = create_adamw_optimizer(model.parameters(), adamw_params)

    assert adamw_opt.param_groups[0]["lr"] == 0.003, "Learning rate not set correctly"
    assert (
        adamw_opt.param_groups[0]["weight_decay"] == 0.1
    ), "Weight decay not set correctly"

    # Test SGD optimizer settings
    sgd_params = {
        "lr": 0.1,
        "momentum": 0.95,
        "nesterov": True,
    }
    sgd_opt = create_sgd_optimizer(model.parameters(), sgd_params)

    assert sgd_opt.param_groups[0]["lr"] == 0.1, "Learning rate not set correctly"
    assert sgd_opt.param_groups[0]["momentum"] == 0.95, "Momentum not set correctly"
    assert (
        sgd_opt.param_groups[0]["nesterov"] == True
    ), "Nesterov flag not set correctly"


def test_optimizer_step():
    """Test basic optimization step."""
    # Create a simple model
    model = nn.Linear(2, 1)
    torch.nn.init.ones_(model.weight)
    torch.nn.init.zeros_(model.bias)

    # Input data
    X = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [0.0]])

    # Test different optimizers
    optimizers = [
        create_adam_optimizer(model.parameters(), {"lr": 0.1}),
        create_adamw_optimizer(model.parameters(), {"lr": 0.1}),
        create_sgd_optimizer(model.parameters(), {"lr": 0.1}),
    ]

    for optimizer in optimizers:
        # Reset model
        torch.nn.init.ones_(model.weight)
        torch.nn.init.zeros_(model.bias)

        # Initial weight sum
        initial_weight_sum = model.weight.sum().item()

        # Forward pass
        y_pred = model(X)
        loss = ((y_pred - y) ** 2).sum()

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Check weights were updated
        current_weight_sum = model.weight.sum().item()
        assert (
            current_weight_sum != initial_weight_sum
        ), f"Weights not updated by {optimizer.__class__.__name__}"
