"""Factory for creating diffusion models with automatic registration."""

from typing import Dict, Callable, Any, Optional, Union, Type
import importlib
import inspect


DIFFUSION_MODEL_REGISTRY: Dict[str, Callable] = {}


def register_diffusion_model(
    model_type: str, variant: Optional[str] = None
) -> Callable:
    """
    Decorator to register a diffusion model constructor function.

    This decorator allows for automatic registration of model constructors
    to the global registry, enabling factory pattern instantiation.

    Args:
        model_type: Base model type (e.g., 'elucidated', 'ddpm')
        variant: Optional variant name (e.g., 'image', 'conditional')

    Returns:
        Decorator function that registers the constructor
    """

    def decorator(constructor_fn: Callable) -> Callable:
        key = f"{model_type}_{variant}" if variant else model_type
        DIFFUSION_MODEL_REGISTRY[key.lower()] = constructor_fn
        return constructor_fn

    return decorator


def create_diffusion_model(config: Dict[str, Any]) -> Any:
    """
    Create diffusion model based on model type and variant specified in config.

    This factory function instantiates the appropriate diffusion model based on
    the configuration dictionary, looking up the constructor in the registry.

    Args:
        config: Configuration dictionary containing model parameters including:
               - model_params.model_type: The type of diffusion model
               - model_params.model_variant: Optional variant specification

    Returns:
        Diffusion model instance of the requested type and variant

    Raises:
        ValueError: If the requested model type/variant is not found in the registry
    """
    # Make sure all models are registered
    _register_all_diffusion_models()

    model_params = config.get("model_params", {})
    model_type = model_params.get("model_type", "elucidated").lower()
    model_variant = model_params.get("model_variant", None)

    # Determine the registry key
    key = f"{model_type}_{model_variant}" if model_variant else model_type
    key = key.lower()

    # Check if the model is registered
    if key in DIFFUSION_MODEL_REGISTRY:
        return DIFFUSION_MODEL_REGISTRY[key](model_params)
    else:
        available_models = ", ".join(DIFFUSION_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unsupported diffusion model type/variant: {key}. Available models: {available_models}"
        )


def _register_all_diffusion_models() -> None:
    """
    Imports all diffusion model modules to ensure all decorated model
    constructor functions are registered with the registry.

    This function is called internally by create_diffusion_model to ensure
    that all model implementations have been properly loaded and registered.
    """
    from . import elucidated_diffusion

    # Add imports for future diffusion model files here
