"""Lazy training entry points with architecture-specific dependencies."""


def train_dino(*args, **kwargs):
    from .dino_trainer import train_dino as implementation

    return implementation(*args, **kwargs)


def train_scot(*args, **kwargs):
    from .scot_trainer import train_scot as implementation

    return implementation(*args, **kwargs)


def train_tfno(*args, **kwargs):
    from .tfno_trainer import train_tfno as implementation

    return implementation(*args, **kwargs)


__all__ = ["train_dino", "train_scot", "train_tfno"]
