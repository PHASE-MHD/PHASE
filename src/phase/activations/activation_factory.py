"""Factory for creating activation functions with automatic registration."""

from typing import Dict, Callable, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# Registry to store activation function constructors
ACTIVATION_REGISTRY: Dict[str, Callable] = {}


def register_activation(activation_type: str):
    """
    Decorator to register an activation function constructor.

    Args:
        activation_type: Name of the activation function

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        ACTIVATION_REGISTRY[activation_type.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_activation(config):
    """
    Create activation function based on type specified in config.

    Args:
        config (dict): Configuration dictionary containing activation parameters

    Returns:
        Activation function or module
    """
    # Make sure all activations are registered
    _register_all_activations()

    # Extract activation config from the full config
    if isinstance(config, str):
        # Allow direct string specification for simple cases
        activation_type = config.lower()
        activation_params = {}
    else:
        # Otherwise, get from activation_params in config
        activation_params = config.get("activation_params", {})
        activation_type = activation_params.get("type", "gelu").lower()

    # Check if the activation is registered
    if activation_type in ACTIVATION_REGISTRY:
        return ACTIVATION_REGISTRY[activation_type](activation_params)
    else:
        available_activations = ", ".join(ACTIVATION_REGISTRY.keys())
        raise ValueError(
            f"Unsupported activation type: {activation_type}. "
            f"Available activations: {available_activations}"
        )


def _register_all_activations():
    """
    Imports all activation modules to ensure all decorated activation
    functions are registered with the registry.
    """
    from . import standard
    from . import advanced
    from . import parameterized
    from . import composed

    # Add imports for future activation files here
