"""Spectral layer implementations for Fourier-based neural operators."""

import torch
import torch.nn as nn
from torch import Tensor
import tltorch


class FactorizedSpectralConv3d(nn.Module):
    """3D Factorized Fourier layer. It does FFT, linear transform, and Inverse FFT.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    modes1 : int
        Number of Fourier modes to multiply in first dimension, at most floor(N/2) + 1
    modes2 : int
        Number of Fourier modes to multiply in second dimension, at most floor(N/2) + 1
    modes3 : int
        Number of Fourier modes to multiply in third dimension, at most floor(N/2) + 1
    rank : float
        Rank of the decomposition
    factorization : {'CP', 'TT', 'Tucker'}
        Tensor factorization to use to decompose the tensor
    fixed_rank_modes : List[int]
        A list of modes for which the initial value is not modified
        The last mode cannot be fixed due to error computation.
    decomposition_kwargs : dict
        Additional arguments to initialization of factorized tensors
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
        rank: float,
        factorization: str,
        fixed_rank_modes: bool,
        decomposition_kwargs: dict,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights2 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights3 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.weights4 = tltorch.FactorizedTensor.new(
            (in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2),
            rank=rank,
            factorization=factorization,
            fixed_rank_modes=fixed_rank_modes,
            **decomposition_kwargs,
        )
        self.reset_parameters()

    def compl_mul3d(self, input: Tensor, weights: Tensor) -> Tensor:
        """Complex multiplication

        Parameters
        ----------
        input : Tensor
            Input tensor
        weights : Tensor
            Weights tensor

        Returns
        -------
        Tensor
            Product of complex multiplication
        """
        # (batch, in_channel, x, y, z), (in_channel, out_channel, x, y, z) -> (batch, out_channel, x, y, z)
        cweights = torch.view_as_complex(weights.to_tensor().contiguous())
        return torch.einsum("bixyz,ioxyz->boxyz", input, cweights)

    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, : self.modes2, : self.modes3], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, : self.modes2, : self.modes3], self.weights2
        )
        out_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, : self.modes1, -self.modes2 :, : self.modes3], self.weights3
        )
        out_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3] = self.compl_mul3d(
            x_ft[:, :, -self.modes1 :, -self.modes2 :, : self.modes3], self.weights4
        )

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x

    def reset_parameters(self):
        """Reset spectral weights with distribution scale*U(0,1)"""
        self.weights1.normal_(0, self.scale)
        self.weights2.normal_(0, self.scale)
        self.weights3.normal_(0, self.scale)
        self.weights4.normal_(0, self.scale)
