"""Composed activation functions that combine multiple activations."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_factory import register_activation, create_activation


class SequentialActivation(nn.Module):
    """
    Sequential application of multiple activation functions.

    Applies a sequence of activation functions in order, with each receiving
    the output of the previous one.
    """

    def __init__(self, activations):
        super().__init__()
        self.activations = nn.ModuleList(activations)

    def forward(self, x):
        for activation in self.activations:
            x = activation(x)
        return x


@register_activation("sequential")
def create_sequential_activation(params):
    """Create a sequential combination of activation functions."""
    activation_configs = params.get("activations", [])
    if not activation_configs:
        raise ValueError(
            "sequential activation requires an 'activations' list parameter"
        )

    activations = []
    for act_config in activation_configs:
        activations.append(create_activation(act_config))

    return SequentialActivation(activations)


class WeightedSumActivation(nn.Module):
    """
    Weighted sum of multiple activation functions.

    Applies multiple activation functions in parallel and combines their
    outputs with learnable weights.
    """

    def __init__(self, activations, initial_weights=None):
        super().__init__()
        self.activations = nn.ModuleList(activations)

        if initial_weights is None:
            initial_weights = torch.ones(len(activations)) / len(activations)

        self.weights = nn.Parameter(initial_weights)

    def forward(self, x):
        # Apply each activation
        outputs = [act(x) for act in self.activations]

        # Get normalized weights
        weights = F.softmax(self.weights, dim=0)

        # Weighted sum
        result = sum(w * out for w, out in zip(weights, outputs))
        return result


@register_activation("weighted_sum")
def create_weighted_sum_activation(params):
    """Create a weighted sum of activation functions."""
    activation_configs = params.get("activations", [])
    if not activation_configs:
        raise ValueError(
            "weighted_sum activation requires an 'activations' list parameter"
        )

    activations = []
    for act_config in activation_configs:
        activations.append(create_activation(act_config))

    initial_weights = params.get("initial_weights")
    if initial_weights:
        initial_weights = torch.tensor(initial_weights, dtype=torch.float32)

    return WeightedSumActivation(activations, initial_weights)


class GatedActivation(nn.Module):
    """
    Gated activation function using two separate activations.

    The first activation processes the input, while the second acts as a gate.

    This generalizes activations like SiLU (x * sigmoid(x)) and Mish (x * tanh(softplus(x))).
    """

    def __init__(self, main_activation, gate_activation):
        super().__init__()
        self.main_activation = main_activation
        self.gate_activation = gate_activation

    def forward(self, x):
        return self.main_activation(x) * self.gate_activation(x)


@register_activation("gated")
def create_gated_activation(params):
    """Create a gated activation function."""
    main_config = params.get("main_activation")
    gate_config = params.get("gate_activation")

    if main_config is None or gate_config is None:
        raise ValueError(
            "gated activation requires both 'main_activation' and 'gate_activation' parameters"
        )

    main_activation = create_activation(main_config)
    gate_activation = create_activation(gate_config)

    return GatedActivation(main_activation, gate_activation)
