"""Tests for the loss modules."""

import pytest
import torch
import torch.nn as nn
from phase.losses.loss_factory import create_loss
from phase.losses.standard import create_mse_loss
from phase.losses.weighted import create_weighted_mse_loss


def test_loss_factory(mini_config):
    """Test loss factory creates the proper loss function."""
    # Create loss from config
    loss = create_loss(mini_config)

    # Check type
    assert isinstance(loss, nn.MSELoss), "Factory should create MSELoss for 'mse' type"

    # Test with different loss types
    weighted_config = {
        "loss_params": {
            "type": "weighted_mse",
            "weights": [1.0, 2.0, 0.5],
            "reduction": "mean",
        }
    }
    weighted_loss = create_loss(weighted_config)
    assert "WeightedMSELoss" in weighted_loss.__class__.__name__, (
        "Factory should create WeightedMSELoss for 'weighted_mse' type"
    )


def test_mse_loss():
    """Test standard MSE loss."""
    # Create MSE loss
    loss = create_mse_loss({"reduction": "mean"})

    # Create tensors
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 3.0], [2.0, 4.0]])

    # Calculate loss
    loss_value = loss(pred, target)

    # Calculate expected loss: mean of (0, 1, 1, 0) = 0.5
    expected = torch.tensor(0.5)

    # Check loss value
    assert torch.isclose(loss_value, expected), (
        f"Expected MSE loss {expected}, got {loss_value}"
    )


def test_weighted_mse_loss():
    """Test weighted MSE loss."""
    # Create weighted MSE loss with channel weights [1.0, 2.0, 0.5]
    loss = create_weighted_mse_loss({"weights": [1.0, 2.0, 0.5], "reduction": "mean"})

    # Create tensors with 3 channels (batch_size=2, channels=3, h=2, w=2)
    pred = torch.ones(2, 3, 2, 2)
    target = torch.ones(2, 3, 2, 2)

    # Add errors to each channel
    pred[:, 0] += 1.0  # Error of 1.0 in channel 0 (weight 1.0)
    pred[:, 1] += 0.5  # Error of 0.5 in channel 1 (weight 2.0)
    pred[:, 2] += 2.0  # Error of 2.0 in channel 2 (weight 0.5)

    # Calculate loss
    loss_value = loss(pred, target)

    # Expected weighted loss:
    # Channel 0: MSE = 1.0² = 1.0, weighted = 1.0 * 1.0 = 1.0
    # Channel 1: MSE = 0.5² = 0.25, weighted = 0.25 * 2.0 = 0.5
    # Channel 2: MSE = 2.0² = 4.0, weighted = 4.0 * 0.5 = 2.0
    # Mean = (1.0 + 0.5 + 2.0) / 3 = 1.167
    expected = torch.tensor(1.167)

    # Check loss value
    assert torch.isclose(loss_value, expected, atol=1e-3), (
        f"Expected weighted MSE loss ~{expected}, got {loss_value}"
    )
