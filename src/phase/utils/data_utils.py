"""Utility functions for handling data normalization and channel management."""

from torch.utils.data import DataLoader
from torch import Tensor
from typing import Tuple, List, Optional, Any


def get_dataset_normalizer(dataloader: DataLoader) -> Optional[Any]:
    """
    Check if the dataset has a normalizer and return it.

    Args:
        dataloader: DataLoader containing the dataset

    Returns:
        The normalizer if it exists, otherwise None
    """
    has_dataset_normalizer = (
        hasattr(dataloader.dataset, "normalizer")
        and dataloader.dataset.normalizer is not None
    )

    if has_dataset_normalizer:
        return dataloader.dataset.normalizer
    else:
        return None


def identify_data_channels(
    inputs: Tensor, target: Tensor
) -> Tuple[List[int], bool, int]:
    """
    Identify data channels in inputs and check for grid embeddings.

    Args:
        inputs: Input tensor with shape [batch, channels, ...]
        target: Target tensor with shape [batch, channels, ...]

    Returns:
        Tuple of (data_channel_indices, has_grid_embeddings, data_channels)
    """
    has_grid_embeddings = inputs.shape[1] > target.shape[1]
    data_channels = target.shape[1]  # Number of data channels

    if has_grid_embeddings:
        data_channel_indices = list(
            range(inputs.shape[1] - data_channels, inputs.shape[1])
        )
    else:
        data_channel_indices = list(range(inputs.shape[1]))

    return data_channel_indices, has_grid_embeddings, data_channels


def apply_denormalization(
    inputs: Tensor,
    target: Tensor,
    pred: Tensor,
    normalizer: Optional[Any],
    data_channel_indices: List[int],
    has_grid_embeddings: bool,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Apply denormalization to inputs, target, and predicted tensors if a normalizer is available.

    Args:
        inputs: Input tensor
        target: Target tensor
        pred: Predicted tensor
        normalizer: Normalizer object with denormalize method
        data_channel_indices: List of indices for data channels in inputs
        has_grid_embeddings: Flag indicating if inputs include grid embeddings

    Returns:
        Tuple of (inputs_denorm, target_denorm, pred_denorm)
    """
    if normalizer is not None and hasattr(normalizer, "denormalize"):
        # Handle separate denormalization for input and output when dimensions differ
        if has_grid_embeddings:
            # For input tensor, we only want to denormalize the data channels, not the grid
            inputs_data_only = inputs[:, data_channel_indices]
            inputs_denorm = normalizer.denormalize(inputs_data_only)
        else:
            # If no grid embeddings, denormalize the whole input
            inputs_denorm = normalizer.denormalize(inputs)

        target_denorm = normalizer.denormalize(target)
        pred_denorm = normalizer.denormalize(pred)
    else:
        # If no normalizer is available, use the data as is
        inputs_denorm = inputs
        target_denorm = target
        pred_denorm = pred

    return inputs_denorm, target_denorm, pred_denorm
