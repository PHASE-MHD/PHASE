"""Factory for creating normalization strategies with automatic registration."""

from typing import Dict, Callable, Any, Optional
import torch
import torch.nn as nn

# Registry to store normalization constructors
NORMALIZATION_REGISTRY: Dict[str, Callable] = {}


def register_normalization(norm_type: str):
    """
    Decorator to register a normalization function constructor.

    Args:
        norm_type: Name of the normalization strategy

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        NORMALIZATION_REGISTRY[norm_type.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_normalization(config):
    """
    Create normalization method based on type specified in config.

    Args:
        config (dict): Configuration dictionary containing normalization parameters

    Returns:
        Normalization function or module
    """
    # Make sure all normalizations are registered
    _register_all_normalizations()

    # Get normalization configuration
    norm_params = config.get("normalization_params", {})
    norm_type = norm_params.get("type", "identity").lower()

    # Handle common naming variations
    if norm_type == "min_max":
        norm_type = "minmax"
    elif norm_type == "standardization":
        norm_type = "standard"
    elif norm_type == "physics_norm":
        norm_type = "physics"

    # Check if the normalization is registered
    if norm_type in NORMALIZATION_REGISTRY:
        return NORMALIZATION_REGISTRY[norm_type](norm_params)
    else:
        available_norms = ", ".join(NORMALIZATION_REGISTRY.keys())
        raise ValueError(
            f"Unsupported normalization type: {norm_type}. Available normalizations: {available_norms}"
        )


def _register_all_normalizations():
    """
    Imports all normalization modules to ensure all decorated normalization
    functions are registered with the registry.
    """
    from . import identity_norm
    from . import physics_norm
    from . import min_max_norm
    from . import standard_norm  # Add this line

    # Add imports for future normalization files here
