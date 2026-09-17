"""Advanced activation functions beyond the standard PyTorch offerings."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation_factory import register_activation


@register_activation("swish")
@register_activation("silu")  # Alias for Swish
def create_swish(params):
    """
    Create Swish activation function (SiLU in PyTorch).

    f(x) = x * sigmoid(x)
    """
    return nn.SiLU(inplace=params.get("inplace", False))


@register_activation("mish")
def create_mish(params):
    """
    Create Mish activation function.

    f(x) = x * tanh(softplus(x))
    """
    return nn.Mish(inplace=params.get("inplace", False))


class Snake(nn.Module):
    """
    Snake activation function.

    f(x) = x + sin²(alpha * x)

    Paper: "Neural Networks Fail to Learn Periodic Functions and How to Fix It"
    https://arxiv.org/abs/2006.08195
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return x + torch.pow(torch.sin(self.alpha * x), 2)


@register_activation("snake")
def create_snake(params):
    """Create Snake activation function."""
    alpha = params.get("alpha", 1.0)
    return Snake(alpha=alpha)


class GEGLU(nn.Module):
    """
    Gated GELU activation.

    From the paper: "GLU Variants Improve Transformer"
    https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
        self.dim_out = dim_out

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


@register_activation("geglu")
def create_geglu(params):
    """Create GEGLU activation (needs dim_in and dim_out)."""
    dim_in = params.get("dim_in")
    dim_out = params.get("dim_out")
    if dim_in is None or dim_out is None:
        raise ValueError("GEGLU requires dim_in and dim_out parameters")
    return GEGLU(dim_in=dim_in, dim_out=dim_out)


class SoftExponential(nn.Module):
    """
    Soft Exponential activation.

    From the paper: "Soft-Exponential: A New Activation Function"
    """

    def __init__(self, alpha=0.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, x):
        # Handle different cases of alpha
        mask_pos = self.alpha >= 0
        mask_neg = self.alpha < 0

        # alpha > 0: exp(alpha * x) - 1 / alpha
        out_pos = (torch.exp(self.alpha * x) - 1) / self.alpha

        # alpha < 0: -ln(1 - alpha * (x + alpha)) / alpha
        out_neg = -torch.log(1 - self.alpha * (x + self.alpha)) / self.alpha

        # alpha = 0: x (linear case, handled implicitly)
        out_zero = x

        # Combine based on alpha value
        return (
            mask_pos.float() * out_pos
            + mask_neg.float() * out_neg
            + (self.alpha == 0).float() * out_zero
        )


@register_activation("soft_exp")
def create_soft_exponential(params):
    """Create Soft Exponential activation function."""
    alpha = params.get("alpha", 0.0)
    return SoftExponential(alpha=alpha)
