"""Training entry points."""

from .dino_trainer import train_dino
from .tfno_trainer import train_tfno

__all__ = ["train_dino", "train_tfno"]
