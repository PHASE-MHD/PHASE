"""Standard PyTorch loss functions wrapped with the registry."""

import torch
import torch.nn as nn
from .loss_factory import register_loss


@register_loss("mse")
def create_mse_loss(params):
    """Create Mean Squared Error loss."""
    reduction = params.get("reduction", "mean")
    return nn.MSELoss(reduction=reduction)


@register_loss("l1")
def create_l1_loss(params):
    """Create L1 (Mean Absolute Error) loss."""
    reduction = params.get("reduction", "mean")
    return nn.L1Loss(reduction=reduction)


@register_loss("smoothl1")
def create_smooth_l1_loss(params):
    """Create Smooth L1 loss."""
    reduction = params.get("reduction", "mean")
    beta = params.get("beta", 1.0)
    return nn.SmoothL1Loss(reduction=reduction, beta=beta)


@register_loss("crossentropy")
def create_cross_entropy_loss(params):
    """Create Cross Entropy loss."""
    reduction = params.get("reduction", "mean")
    weight = params.get("weight", None)
    if weight is not None:
        weight = torch.tensor(weight)
    return nn.CrossEntropyLoss(weight=weight, reduction=reduction)
