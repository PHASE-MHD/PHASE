"""Factory for creating optimizers with automatic registration."""

from typing import Dict, Callable, Any, Optional, Iterator

# Registry to store optimizer constructors
OPTIMIZER_REGISTRY: Dict[str, Callable] = {}


def register_optimizer(optimizer_type: str):
    """
    Decorator to register an optimizer constructor function.

    Args:
        optimizer_type: Name of the optimizer

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        OPTIMIZER_REGISTRY[optimizer_type.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_optimizer(parameters, config):
    """
    Create optimizer based on type specified in config.

    Args:
        parameters: Model parameters to optimize
        config (dict): Configuration dictionary containing optimizer parameters

    Returns:
        Optimizer instance of the requested type
    """
    # Get optimizer configuration
    optimizer_params = config.get("optimizer_params", {})
    optimizer_type = optimizer_params.get("optimizer_type", "adam").lower()

    # Make sure all optimizers are registered
    _register_all_optimizers()

    # Check if the optimizer is registered
    if optimizer_type in OPTIMIZER_REGISTRY:
        return OPTIMIZER_REGISTRY[optimizer_type](parameters, optimizer_params)
    else:
        available_optimizers = ", ".join(OPTIMIZER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported optimizer type: {optimizer_type}. Available optimizers: {available_optimizers}"
        )


def _register_all_optimizers():
    """
    Imports all modules in the optimizers package to ensure all decorated optimizer
    functions are registered with the registry.
    """
    # Import all optimizer modules
    from . import standard_optimizers
