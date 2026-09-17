"""Factory for creating loss functions with support for custom implementations."""

from typing import Dict, Callable, Any, Optional
import importlib
import pkgutil
import inspect
import sys

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
    loss_type = loss_config.get("type", "mse").lower()

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
    # Import all modules in this package
    from . import standard
    from . import weighted
    from . import physics_informed

    # For future expansion, you can also automatically discover and import all modules
    # in the losses package using this pattern:
    """
    import importlib
    import pkgutil
    import sys

    # Dynamically import all modules in the current package
    current_package = sys.modules[__package__]
    for _, name, is_pkg in pkgutil.iter_modules(current_package.__path__, current_package.__name__ + '.'):
        if not is_pkg:  # Only import modules, not sub-packages
            importlib.import_module(name)
    """
