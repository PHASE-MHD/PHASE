"""Explicit model construction for the public PHASE runtime."""

from __future__ import annotations

from collections.abc import Mapping


def create_model(config: Mapping):
    """Create the model selected by ``config['model_params']``."""
    params = dict(config["model_params"])
    model_type = str(params.pop("model_type", "tfno")).lower()
    variant = params.pop("model_variant", None)

    if model_type == "tfno" and variant in (None, "3d"):
        from .tfno import TFNO

        params["dimension"] = 3
        return TFNO(**params)

    if model_type in {"poseidon-mhd-finetune", "scot-mhd"} and variant is None:
        from .scot_mhd import create_poseidon_mhd_finetune

        return create_poseidon_mhd_finetune(params)

    if model_type == "poseidon-mhd-re-input-finetune" and variant is None:
        from .scot_mhd_naive_re import create_poseidon_mhd_re_input_finetune

        return create_poseidon_mhd_re_input_finetune(params)

    if model_type == "poseidon-mhd-re-finetune" and variant is None:
        from .scot_mhd_gated_re import create_poseidon_mhd_re_finetune

        return create_poseidon_mhd_re_finetune(params)

    raise ValueError(
        f"Unsupported model_type={model_type!r}, model_variant={variant!r}. "
        "Supported models are tfno/3d, poseidon-mhd-finetune, and "
        "poseidon-mhd-re-input-finetune, and poseidon-mhd-re-finetune."
    )
