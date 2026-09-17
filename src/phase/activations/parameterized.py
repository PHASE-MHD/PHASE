"""Parameterized activation functions with learnable parameters."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_factory import register_activation


@register_activation("leaky_relu")
def create_leaky_relu(params):
    """Create Leaky ReLU activation function."""
    negative_slope = params.get("negative_slope", 0.01)
    return nn.LeakyReLU(
        negative_slope=negative_slope, inplace=params.get("inplace", False)
    )


@register_activation("prelu")
def create_prelu(params):
    """Create PReLU activation function with learnable parameters."""
    num_parameters = params.get("num_parameters", 1)
    init = params.get("init", 0.25)
    return nn.PReLU(num_parameters=num_parameters, init=init)


@register_activation("rrelu")
def create_rrelu(params):
    """Create Randomized Leaky ReLU activation function."""
    lower = params.get("lower", 1 / 8)
    upper = params.get("upper", 1 / 3)
    return nn.RReLU(lower=lower, upper=upper, inplace=params.get("inplace", False))


class PELU(nn.Module):
    """
    Parametric Exponential Linear Unit.

    From the paper: "PELU: Parametric Exponential Linear Unit for Deep Learning"
    https://arxiv.org/abs/1605.09332

    f(x) = x if x > 0, alpha * (exp(x/beta) - 1) if x <= 0
    """

    def __init__(self, alpha_init=1.0, beta_init=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))

    def forward(self, x):
        pos = F.relu(x)
        neg = self.alpha * (
            torch.exp(torch.min(torch.zeros_like(x), x) / self.beta) - 1
        )
        return pos + neg


@register_activation("pelu")
def create_pelu(params):
    """Create PELU activation function with learnable parameters."""
    alpha_init = params.get("alpha_init", 1.0)
    beta_init = params.get("beta_init", 1.0)
    return PELU(alpha_init=alpha_init, beta_init=beta_init)


class AdaptiveActivation(nn.Module):
    """
    Adaptive activation function with learnable parameters.

    Learns different activation functions for different channels.
    A linear combination of common activation functions.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_channels, 5))
        self.weights.data = F.softmax(self.weights, dim=1)

    def forward(self, x):
        # Extract batch and channel dimensions
        batch_size, num_channels = x.shape[0], x.shape[1]

        # Compute different activations
        act_functions = [
            x,  # identity
            torch.tanh(x),  # tanh
            F.relu(x),  # relu
            torch.sigmoid(x),  # sigmoid
            F.silu(x),  # swish
        ]

        # Reshape weights for broadcasting
        weights = F.softmax(self.weights, dim=1).view(
            1, num_channels, 5, *([1] * (x.dim() - 2))
        )

        # Stack activations for weighted sum
        acts = torch.stack(act_functions, dim=2)

        # Perform weighted sum across activation dimension
        return (acts * weights).sum(dim=2)


@register_activation("adaptive")
def create_adaptive_activation(params):
    """Create adaptive activation function with learnable weights."""
    num_channels = params.get("num_channels")
    if num_channels is None:
        raise ValueError("adaptive activation requires num_channels parameter")
    return AdaptiveActivation(num_channels=num_channels)
