"""Utility functions for tensor denormalization with channel mapping."""

import torch


def denormalize_with_channel_mapping(tensor, normalizer, channel_indices, re=None):
    """
    Denormalizes tensors with proper channel mapping to handle different channel counts.

    This function handles the case where normalizer expects a specific number of channels
    (typically 3) but the input tensor may have fewer channels.

    Args:
        tensor: Input tensor to denormalize with shape [batch_size, channels, nx, ny]
        normalizer: Normalizer object with denormalize method
        channel_indices: Indices specifying which channels in normalizer space the tensor channels map to
        re: Optional Re metadata for Re-dependent normalizers

    Returns:
        Denormalized tensor with same shape as input
    """
    if normalizer is None:
        return tensor

    # Get tensor shape
    bs, c, *spatial_dims = tensor.shape

    # First add the singleton time dimension required by the normalizer
    tensor_5d = tensor.unsqueeze(2)  # (bs, c, 1, nx, ny)

    # Create a full 3-channel tensor with zeros
    full_tensor = torch.zeros(bs, max(channel_indices) + 1, 1, *spatial_dims, device=tensor.device)

    # Place each channel in its correct position according to channel_indices
    for i, channel_idx in enumerate(channel_indices):
        if i < c:  # Only use as many channels as we have in our tensor
            full_tensor[:, channel_idx] = tensor_5d[:, i]

    # Apply denormalization to the full tensor
    try:
        denormalized = normalizer.denormalize(full_tensor, re=re)
    except TypeError:
        denormalized = normalizer.denormalize(full_tensor)

    # Extract only the channels we need based on original indices
    result = torch.zeros(bs, c, *spatial_dims, device=tensor.device)
    for i, channel_idx in enumerate(channel_indices):
        if i < c:  # Only extract as many channels as we need
            result[:, i] = denormalized[:, channel_idx, 0]  # Remove time dimension

    return result


def normalize_with_channel_mapping(tensor, normalizer, channel_indices, re=None):
    """
    Normalizes tensors with the same channel mapping used for denormalization.

    Args:
        tensor: Physical-unit tensor with shape [batch_size, channels, nx, ny]
        normalizer: Normalizer object with normalize method
        channel_indices: Indices specifying which channels in normalizer space the tensor maps to
        re: Optional Re metadata for Re-dependent normalizers

    Returns:
        Normalized tensor with same shape as input
    """
    if normalizer is None:
        return tensor

    bs, c, *spatial_dims = tensor.shape
    tensor_5d = tensor.unsqueeze(2)
    full_tensor = torch.zeros(
        bs,
        max(channel_indices) + 1,
        1,
        *spatial_dims,
        device=tensor.device,
        dtype=tensor.dtype,
    )

    for i, channel_idx in enumerate(channel_indices):
        if i < c:
            full_tensor[:, channel_idx] = tensor_5d[:, i]

    try:
        normalized = normalizer.normalize(full_tensor, re=re)
    except TypeError:
        normalized = normalizer.normalize(full_tensor)

    result = torch.zeros(bs, c, *spatial_dims, device=tensor.device, dtype=tensor.dtype)
    for i, channel_idx in enumerate(channel_indices):
        if i < c:
            result[:, i] = normalized[:, channel_idx, 0]

    return result


def project_residual_via_full_field(
    inputs,
    residual,
    input_normalizer,
    target_normalizer,
    channel_indices,
    projection_module,
    re=None,
):
    """
    Project a residual prediction by first reconstructing the full physical field.

    The residual diffusion target is normalized as (truth - scOT). Projecting that
    residual directly is incorrect because the divergence-free and mean-mode
    constraints apply to the full field. This helper:

      1. denormalizes the scOT condition and residual prediction,
      2. forms full_pred = scOT + residual,
      3. applies Helmholtz projection to full_pred,
      4. converts back to a residual relative to the unprojected scOT condition,
      5. normalizes the projected residual with the target normalizer.
    """
    if input_normalizer is None or target_normalizer is None:
        raise ValueError("Full-field residual projection requires both normalizers.")
    if channel_indices is None:
        channel_indices = list(range(inputs.shape[1]))

    input_field = denormalize_with_channel_mapping(
        inputs, input_normalizer, channel_indices, re=re
    )
    residual_field = denormalize_with_channel_mapping(
        residual, target_normalizer, channel_indices, re=re
    )
    projected_full = projection_module(input_field + residual_field)
    projected_residual = projected_full - input_field
    return normalize_with_channel_mapping(
        projected_residual, target_normalizer, channel_indices, re=re
    )


def is_residual_prediction_mode(config):
    """Return True when diffusion samples represent residual corrections."""
    dataset_params = (config or {}).get("dataset_params", {})
    mode = str(dataset_params.get("prediction_mode", "direct")).lower()
    return bool(dataset_params.get("residual_target", mode == "residual"))


def reconstruct_residual_prediction(
    inputs,
    pred,
    target,
    input_normalizer=None,
    target_normalizer=None,
    channel_indices=None,
    re=None,
    denormalize=True,
    projection_module=None,
    project_full_field=False,
):
    """Convert residual diffusion outputs to physical field predictions.

    In residual mode the diffusion model predicts delta = truth - conditional_input.
    Metrics and plots should therefore compare input + delta against truth.  When
    normalizers are provided, reconstruction is done after denormalizing input and
    residual tensors with their respective normalizers.
    """
    if channel_indices is None:
        channel_indices = list(range(inputs.shape[1]))

    if denormalize and input_normalizer is not None and target_normalizer is not None:
        input_field = denormalize_with_channel_mapping(
            inputs, input_normalizer, channel_indices, re=re
        )
        pred_residual = denormalize_with_channel_mapping(
            pred, target_normalizer, channel_indices, re=re
        )
        target_residual = denormalize_with_channel_mapping(
            target, target_normalizer, channel_indices, re=re
        )
    else:
        input_field = inputs
        pred_residual = pred
        target_residual = target

    pred_field = input_field + pred_residual
    target_field = input_field + target_residual
    if project_full_field and projection_module is not None:
        pred_field = projection_module(pred_field)
    return pred_field, target_field, input_field
