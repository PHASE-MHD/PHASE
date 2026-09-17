"""Normalization strategies for neural operator models."""

from .normalization_factory import (
    create_normalization,
    register_normalization,
    NORMALIZATION_REGISTRY,
)

from .physics_norm import PhysicsNormalization

__all__ = [
    "create_normalization",
    "register_normalization",
    "NORMALIZATION_REGISTRY",
    "PhysicsNormalization",
]
