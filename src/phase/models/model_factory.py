"""Explicit model construction for the public PHASE runtime."""

from __future__ import annotations

from collections.abc import Mapping


def create_model(config: Mapping):
    """Create the model selected by ``config['model_params']``."""
    params = dict(config["model_params"])
    model_type = str(params.pop("model_type", "tfno")).lower()
    variant = params.pop("model_variant", None)

    if model_type != "tfno" or variant not in (None, "3d"):
        raise ValueError(
            "This runtime supports only model_type='tfno' with model_variant='3d'."
        )

    from .tfno import TFNO

    params["dimension"] = 3
    return TFNO(**params)
