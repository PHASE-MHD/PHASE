"""Weighted loss implementations that apply different weights to different components."""

import torch
import torch.nn as nn
from .loss_factory import register_loss


@register_loss("weighted_mse")
def create_weighted_mse_loss(params):
    """Create a custom weighted MSE loss that applies different weights to different components."""
    weights = params.get("weights", [1.0, 1.0, 1.0])  # Weights for different components
    reduction = params.get("reduction", "mean")

    class WeightedMSELoss(nn.Module):
        def __init__(self, weights, reduction="mean"):
            super().__init__()
            self.weights = torch.tensor(weights)
            self.reduction = reduction

        def forward(self, pred, target):
            # Assuming pred and target have shape [batch, channels, *]
            # and weights has length equal to number of channels
            self.weights = self.weights.to(pred.device)

            squared_diff = (pred - target) ** 2

            if squared_diff.dim() > 2:  # Handle spatial dimensions
                # Compute mean over spatial dimensions for each channel
                channel_mse = squared_diff.mean(dim=tuple(range(2, squared_diff.dim())))
                weighted_mse = (channel_mse * self.weights.view(1, -1)).mean(dim=1)
            else:
                weighted_mse = (squared_diff * self.weights).mean(dim=1)

            if self.reduction == "mean":
                return weighted_mse.mean()
            elif self.reduction == "sum":
                return weighted_mse.sum()
            else:  # 'none'
                return weighted_mse

    return WeightedMSELoss(weights, reduction)


@register_loss("weighted_l1")
def create_weighted_l1_loss(params):
    """Create a custom weighted L1 loss that applies different weights to different components."""
    weights = params.get("weights", [1.0, 1.0, 1.0])
    reduction = params.get("reduction", "mean")

    class WeightedL1Loss(nn.Module):
        def __init__(self, weights, reduction="mean"):
            super().__init__()
            self.weights = torch.tensor(weights)
            self.reduction = reduction

        def forward(self, pred, target):
            self.weights = self.weights.to(pred.device)

            abs_diff = torch.abs(pred - target)

            if abs_diff.dim() > 2:
                channel_l1 = abs_diff.mean(dim=tuple(range(2, abs_diff.dim())))
                weighted_l1 = (channel_l1 * self.weights.view(1, -1)).mean(dim=1)
            else:
                weighted_l1 = (abs_diff * self.weights).mean(dim=1)

            if self.reduction == "mean":
                return weighted_l1.mean()
            elif self.reduction == "sum":
                return weighted_l1.sum()
            else:  # 'none'
                return weighted_l1

    return WeightedL1Loss(weights, reduction)
