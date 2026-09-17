"""Loss functions for training and evaluation."""

from typing import Union, Optional, Tuple
import torch
import torch.nn as nn
from torch import Tensor


class LpLoss:
    """
    Loss function with relative/absolute Lp loss.

    Parameters
    ----------
    d : int
        Dimension of the domain
    p : int
        Order of the norm
    size_average : bool
        Whether to average over batch
    reduction : bool
        Whether to average/sum over error
    """

    def __init__(
        self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True
    ):
        # Dimension and Lp-norm type are positive
        assert d > 0 and p > 0, "Dimension and norm order must be positive"

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x: Tensor, y: Tensor) -> Union[Tensor, float]:
        """
        Compute absolute Lp loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            Absolute Lp loss
        """
        num_examples = x.size()[0]

        # Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h ** (self.d / self.p)) * torch.norm(
            x.view(num_examples, -1) - y.view(num_examples, -1), self.p, 1
        )

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x: Tensor, y: Tensor) -> Union[Tensor, float]:
        """
        Compute relative Lp loss.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            Relative Lp loss
        """
        num_examples = x.size()[0]

        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1
        )
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            else:
                return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x: Tensor, y: Tensor) -> Union[Tensor, float]:
        """
        Compute relative Lp loss by default.

        Args:
            x: Predicted tensor
            y: Target tensor

        Returns:
            Lp loss (relative by default)
        """
        return self.rel(x, y)
