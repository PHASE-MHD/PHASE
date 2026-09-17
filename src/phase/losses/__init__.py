"""Loss functions and factory for creating them from configuration."""

from .loss_factory import LOSS_REGISTRY, create_loss, register_loss
from .lp_loss import LpLoss
from .physics_informed import MHDVecPotLoss

__all__ = [
    "LOSS_REGISTRY",
    "LpLoss",
    "MHDVecPotLoss",
    "create_loss",
    "register_loss",
]
