"""Tests for the scheduler modules."""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from phase.optimizers.scheduler_factory import create_scheduler
from phase.optimizers.standard_schedulers import (
    create_dummy_scheduler,
    create_multistep_scheduler,
    create_cosine_scheduler,
    create_reduce_on_plateau_scheduler,
)


def test_scheduler_factory(mini_config):
    """Test scheduler factory creates the proper scheduler."""
    # Create a simple model for optimizer
    model = nn.Linear(10, 1)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Update config to specify scheduler type
    config = mini_config.copy()
    config["optimizer_params"] = {
        "scheduler_type": "multistep",
        "milestones": [10, 20],
        "gamma": 0.5,
        "use_scheduler": True,
    }

    # Create scheduler from config
    scheduler = create_scheduler(optimizer, config)

    # Check type
    assert isinstance(scheduler, optim.lr_scheduler.MultiStepLR), (
        "Factory should create MultiStepLR scheduler for 'multistep' type"
    )

    # Test with different scheduler types
    cosine_config = {
        "optimizer_params": {
            "scheduler_type": "cosine",
            "T_max": 100,
            "use_scheduler": True,
        }
    }
    cosine_scheduler = create_scheduler(optimizer, cosine_config)
    assert isinstance(cosine_scheduler, optim.lr_scheduler.CosineAnnealingLR), (
        "Factory should create CosineAnnealingLR for 'cosine' type"
    )

    reducelr_config = {
        "optimizer_params": {
            "scheduler_type": "reducelr",
            "mode": "min",
            "use_scheduler": True,
        }
    }
    reducelr_scheduler = create_scheduler(optimizer, reducelr_config)
    assert isinstance(reducelr_scheduler, optim.lr_scheduler.ReduceLROnPlateau), (
        "Factory should create ReduceLROnPlateau for 'reducelr' type"
    )

    dummy_config = {"optimizer_params": {"use_scheduler": False}}
    dummy_scheduler = create_scheduler(optimizer, dummy_config)
    assert isinstance(dummy_scheduler, optim.lr_scheduler.LambdaLR), (
        "Factory should create LambdaLR for 'dummy' type"
    )


def test_multistep_scheduler():
    """Test MultiStepLR scheduler."""
    # Create a simple model
    model = nn.Linear(10, 1)
    input_data = torch.randn(1, 10)
    target = torch.randn(1, 1)
    criterion = nn.MSELoss()

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create scheduler
    scheduler = create_multistep_scheduler(
        optimizer, {"milestones": [1, 2], "gamma": 0.1}
    )

    # Check initial learning rate
    assert optimizer.param_groups[0]["lr"] == 0.001, "Initial LR should be 0.001"

    # Simulate one training iteration before stepping scheduler
    output = model(input_data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step the scheduler after optimizer (proper order)
    scheduler.step()

    # Check new learning rate
    assert optimizer.param_groups[0]["lr"] == 0.0001, "LR should be reduced to 0.0001"

    # Simulate another iteration
    output = model(input_data)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Step again
    scheduler.step()

    # Check final learning rate
    assert optimizer.param_groups[0]["lr"] == 0.00001, "LR should be reduced to 0.00001"


def test_cosine_scheduler():
    """Test CosineAnnealingLR scheduler."""
    # Create a simple model
    model = nn.Linear(10, 1)
    input_data = torch.randn(1, 10)
    target = torch.randn(1, 1)
    criterion = nn.MSELoss()

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create scheduler
    scheduler = create_cosine_scheduler(optimizer, {"T_max": 10, "eta_min": 0})

    # Check initial learning rate
    assert optimizer.param_groups[0]["lr"] == 0.001, "Initial LR should be 0.001"

    # Step the scheduler multiple times and check LR decreases
    initial_lr = optimizer.param_groups[0]["lr"]

    for _ in range(5):
        # Simulate training step
        output = model(input_data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update scheduler
        scheduler.step()

    # Check that LR has decreased
    assert optimizer.param_groups[0]["lr"] < initial_lr, (
        "LR should decrease with cosine scheduler"
    )

    # Continue stepping to T_max
    for _ in range(5):
        # Simulate training step
        output = model(input_data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update scheduler
        scheduler.step()

    # Check that LR is at minimum
    assert optimizer.param_groups[0]["lr"] == 0, (
        "LR should reach eta_min after T_max steps"
    )


def test_reducelr_scheduler():
    """Test ReduceLROnPlateau scheduler."""
    # Create a simple model
    model = nn.Linear(10, 1)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create scheduler with small patience for testing
    scheduler = create_reduce_on_plateau_scheduler(
        optimizer, {"mode": "min", "factor": 0.1, "patience": 1, "threshold": 1e-4}
    )

    # Check initial learning rate
    assert optimizer.param_groups[0]["lr"] == 0.001, "Initial LR should be 0.001"

    # For ReduceLROnPlateau, no need for optimizer.step() before scheduler.step(metric)
    # as it's monitoring the metric value, not tied to training iterations

    # Simulate no improvement for patience+1 epochs
    scheduler.step(1.0)  # First validation loss
    scheduler.step(1.0)  # No improvement, patience not exhausted
    scheduler.step(1.0)  # No improvement, patience exhausted, should reduce LR

    # Check new learning rate
    assert optimizer.param_groups[0]["lr"] == 0.0001, "LR should be reduced to 0.0001"

    # Simulate improvement (metric goes down)
    scheduler.step(0.5)

    # LR shouldn't change when improving
    assert optimizer.param_groups[0]["lr"] == 0.0001, (
        "LR should remain the same after improvement"
    )


def test_dummy_scheduler():
    """Test dummy scheduler (no changes to LR)."""
    # Create a simple model
    model = nn.Linear(10, 1)
    input_data = torch.randn(1, 10)
    target = torch.randn(1, 1)
    criterion = nn.MSELoss()

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Create scheduler
    scheduler = create_dummy_scheduler(optimizer, {})

    # Check initial learning rate
    initial_lr = optimizer.param_groups[0]["lr"]

    # Step the scheduler multiple times
    for _ in range(10):
        # Simulate training step
        output = model(input_data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update scheduler
        scheduler.step()

    # Check that LR hasn't changed
    assert optimizer.param_groups[0]["lr"] == initial_lr, (
        "Dummy scheduler should not change LR"
    )
