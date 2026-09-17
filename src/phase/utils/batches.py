"""Utilities for batches that optionally carry physical-regime metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """Return inputs, targets, and optional metadata from a loader batch."""
    if len(batch) == 2:
        inputs, target = batch
        return inputs, target, None
    if len(batch) == 3:
        inputs, target, metadata = batch
        return inputs, target, metadata
    raise ValueError(f"Expected batch of length 2 or 3, got {len(batch)}.")


def move_metadata_to_device(metadata, device):
    """Move tensor-valued metadata to the training device."""
    if metadata is None:
        return None
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in metadata.items()
    }


def model_forward(model, inputs, metadata=None):
    """Forward metadata only to models that explicitly accept conditioning."""
    if metadata is not None and getattr(model, "accepts_conditioning", False):
        return model(inputs, re=metadata.get("re"), rem=metadata.get("rem"))
    return model(inputs)
