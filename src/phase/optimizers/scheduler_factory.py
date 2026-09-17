"""Factory for creating learning rate schedulers with automatic registration."""

from typing import Dict, Callable, Any, Optional

# Registry to store scheduler constructors
SCHEDULER_REGISTRY: Dict[str, Callable] = {}


def register_scheduler(scheduler_type: str):
    """
    Decorator to register a scheduler constructor function.

    Args:
        scheduler_type: Name of the scheduler

    Returns:
        Decorator function
    """

    def decorator(constructor_fn):
        SCHEDULER_REGISTRY[scheduler_type.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_scheduler(optimizer, config):
    """
    Create scheduler based on type specified in config.

    Args:
        optimizer: The optimizer to schedule
        config (dict): Configuration dictionary containing optimizer parameters

    Returns:
        Scheduler instance of the requested type
    """
    # Get optimizer configuration
    optimizer_params = config.get("optimizer_params", {})

    # Populate the registry before either the enabled or disabled lookup.
    _register_all_schedulers()

    # Check if scheduler is enabled
    if not optimizer_params.get("use_scheduler", True):
        # Return the dummy scheduler if scheduling is disabled
        return SCHEDULER_REGISTRY["dummy"](optimizer, optimizer_params)

    scheduler_type = optimizer_params.get("scheduler_type", "multistep").lower()

    # Check if the scheduler is registered
    if scheduler_type in SCHEDULER_REGISTRY:
        return SCHEDULER_REGISTRY[scheduler_type](optimizer, optimizer_params)
    else:
        available_schedulers = ", ".join(SCHEDULER_REGISTRY.keys())
        raise ValueError(
            f"Unsupported scheduler type: {scheduler_type}. Available schedulers: {available_schedulers}"
        )


def _register_all_schedulers():
    """
    Imports all modules in the optimizers package to ensure all decorated scheduler
    functions are registered with the registry.
    """
    # Import all scheduler modules
    from . import standard_schedulers
