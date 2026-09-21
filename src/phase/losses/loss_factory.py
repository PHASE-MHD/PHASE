"""Factory for creating loss functions with support for custom implementations."""

from typing import Callable, Dict

# Registry to store loss function constructors
LOSS_REGISTRY: Dict[str, Callable] = {}


def register_loss(loss_type: str):
    """
    Decorator to register a loss function constructor.

    Args:
        loss_type: Name of the loss function

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        LOSS_REGISTRY[loss_type.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_loss(config):
    """
    Create loss function based on type specified in config.

    Args:
        config (dict): Configuration dictionary containing loss parameters

    Returns:
        Loss function instance
    """
    loss_config = config.get("loss_params", {})
    loss_type = loss_config.get("type")
    if not loss_type:
        raise ValueError("loss_params.type is required.")
    loss_type = loss_type.lower()

    # Make sure all losses are registered before checking
    _register_all_losses()

    # Check if the loss is registered
    if loss_type in LOSS_REGISTRY:
        return LOSS_REGISTRY[loss_type](loss_config)
    else:
        available_losses = ", ".join(LOSS_REGISTRY.keys())
        raise ValueError(
            f"Unsupported loss type: {loss_type}. Available losses: {available_losses}"
        )


def _register_all_losses():
    """
    Imports all modules in the losses package to ensure all decorated loss
    functions are registered with the registry.
    """
    from . import physics_informed
