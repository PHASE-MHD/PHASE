"""Standard PyTorch activation functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_factory import register_activation


@register_activation("relu")
def create_relu(params):
    """Create ReLU activation function."""
    return nn.ReLU(inplace=params.get("inplace", False))


@register_activation("gelu")
def create_gelu(params):
    """Create GELU activation function."""
    approximate = params.get("approximate", "none")
    if approximate == "tanh":
        return nn.GELU(approximate="tanh")
    return nn.GELU()


@register_activation("tanh")
def create_tanh(params):
    """Create Tanh activation function."""
    return nn.Tanh()


@register_activation("sigmoid")
def create_sigmoid(params):
    """Create Sigmoid activation function."""
    return nn.Sigmoid()


@register_activation("softplus")
def create_softplus(params):
    """Create Softplus activation function."""
    beta = params.get("beta", 1)
    threshold = params.get("threshold", 20)
    return nn.Softplus(beta=beta, threshold=threshold)


@register_activation("identity")
def create_identity(params):
    """Create identity (no-op) activation function."""
    return nn.Identity()


@register_activation("elu")
def create_elu(params):
    """Create ELU activation function."""
    alpha = params.get("alpha", 1.0)
    return nn.ELU(alpha=alpha, inplace=params.get("inplace", False))


@register_activation("selu")
def create_selu(params):
    """Create SELU activation function."""
    return nn.SELU(inplace=params.get("inplace", False))
