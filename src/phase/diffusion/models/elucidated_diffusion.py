from math import sqrt
from typing import Tuple, Optional, List, Union, Dict, Any, Callable
import numpy as np
import torch
from torch import nn, einsum
import torch.nn.functional as F

from tqdm import tqdm
from einops import rearrange, repeat, reduce

from ..utils import (
    exists,
    default,
    log,
)
from .diffusion_factory import register_diffusion_model
from .backbones.unet import UNet
from .helmholtz_projection import HelmholtzProjectionDiffusion
from ...utils.diffusion_tensor_normalization import project_residual_via_full_field


class ElucidatedDiffusion(nn.Module):
    """
    Elucidated Diffusion model implementation.

    This model uses a UNet backbone for the denoising process and implements
    the sampling and training process as described in the Elucidated Diffusion paper.
    It provides both regular sampling and DPM++ sampling methods for generating high-quality
    outputs from noise.

    References:
        - Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models"
    """

    def __init__(
        self,
        net: nn.Module,
        *,
        image_size: int,
        channels: int = 3,
        num_sample_steps: int = 32,
        sigma_min: float = 0.002,
        sigma_max: float = 80,
        sigma_data: float = 0.5,
        rho: float = 7,
        P_mean: float = -1.2,
        P_std: float = 1.2,
        S_churn: float = 80,
        S_tmin: float = 0.05,
        S_tmax: float = 50,
        S_noise: float = 1.003,
        helmholtz_projection: bool = False,
        project_velocity: bool = True,
        project_B: bool = True,
        projection_mode: str = "output",
        domain_size_x: float = 1.0,
        domain_size_y: float = 1.0,
        use_vorticity_loss: bool = False,
        use_current_loss: bool = False,
        vorticity_loss_weight: float = 1.0,
        current_loss_weight: float = 5.0,
    ):
        """
        Initialize the Elucidated Diffusion model.

        Args:
            net: Neural network model (typically UNet) for denoising
            image_size: Size of the input/output images (assumes square images)
            channels: Number of channels in the input/output images
            num_sample_steps: Number of discretized steps used during sampling
            sigma_min: Minimum noise level σ_min
            sigma_max: Maximum noise level σ_max
            sigma_data: Standard deviation of (the normalized) data distribution
            rho: Controls the sampling schedule shape
            P_mean: Mean of log-normal noise distribution for training
            P_std: Standard deviation of log-normal noise distribution for training
            S_churn: Stochastic sampling parameter (Table 5 in paper)
            S_tmin: Minimum threshold for applying stochasticity
            S_tmax: Maximum threshold for applying stochasticity
            S_noise: Noise scale for stochastic sampling
        """
        super().__init__()
        self.self_condition = net.self_condition
        self.net = net

        self.channels = channels
        self.image_size = image_size

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        self.num_sample_steps = num_sample_steps
        self.S_churn = S_churn
        self.S_tmin = S_tmin
        self.S_tmax = S_tmax
        self.S_noise = S_noise
        self.domain_size_x = domain_size_x
        self.domain_size_y = domain_size_y
        self.use_vorticity_loss = use_vorticity_loss
        self.use_current_loss = use_current_loss
        self.vorticity_loss_weight = vorticity_loss_weight
        self.current_loss_weight = current_loss_weight
        self.projection_mode = str(projection_mode or "output").lower()

        # Helmholtz projection setup
        self.helmholtz_projection = helmholtz_projection
        if self.helmholtz_projection:
            self.projection_module = HelmholtzProjectionDiffusion(
                Lx=domain_size_x,
                Ly=domain_size_y,
                project_velocity=project_velocity,
                project_B=project_B
            )

    @property
    def device(self) -> torch.device:
        """Get the device on which the model parameters are stored."""
        return next(self.net.parameters()).device

    def get_optimizer_parameters(self, optimizer_params):
        """Return the diffusion parameters as one optimizer group."""
        return self.parameters()

    def c_skip(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the skip connection strength c_skip(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The skip connection coefficient tensor
        """
        return (self.sigma_data**2) / (sigma**2 + self.sigma_data**2)

    def c_out(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the output scaling factor c_out(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The output scaling coefficient tensor
        """
        return sigma * self.sigma_data * (self.sigma_data**2 + sigma**2) ** -0.5

    def c_in(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the input scaling factor c_in(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The input scaling coefficient tensor
        """
        return 1 * (sigma**2 + self.sigma_data**2) ** -0.5

    def c_noise(self, sigma: Union[torch.Tensor, float]) -> torch.Tensor:
        """
        Calculate the noise level embedding c_noise(σ).

        From equation in Table 1 of the paper.

        Args:
            sigma: Noise level

        Returns:
            The noise level embedding tensor
        """
        return log(sigma) * 0.25

    def preconditioned_network_forward(
        self,
        noised_images: torch.Tensor,
        sigma: Union[torch.Tensor, float],
        self_cond: Optional[torch.Tensor] = None,
        clamp: bool = False,
    ) -> torch.Tensor:
        """
        Apply the preconditioned network forward pass.

        Implements equation (7) from the paper: D_θ(x, σ) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x, c_noise(σ))

        Args:
            noised_images: Input noisy images
            sigma: Noise level(s)
            self_cond: Optional self-conditioning input
            clamp: Whether to clamp outputs to [-1, 1]

        Returns:
            Denoised image prediction
        """
        batch, device = noised_images.shape[0], noised_images.device

        if isinstance(sigma, float):
            sigma = torch.full((batch,), sigma, device=device)

        padded_sigma = rearrange(sigma, "b -> b 1 1 1")

        net_out = self.net(
            self.c_in(padded_sigma) * noised_images,
            self.c_noise(sigma),
            self_cond,
        )

        out = (
            self.c_skip(padded_sigma) * noised_images
            + self.c_out(padded_sigma) * net_out
        )

        if clamp:
            out = out.clamp(-1.0, 1.0)

        return out

    def sample_schedule(self, num_sample_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate the noise level schedule for sampling.

        Implements equation (5) from the paper for discretized noise levels.

        Args:
            num_sample_steps: Number of sampling steps (default: self.num_sample_steps)

        Returns:
            Tensor of sigma values for each sampling step
        """
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        N = num_sample_steps
        inv_rho = 1 / self.rho

        steps = torch.arange(num_sample_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.sigma_max**inv_rho
            + steps / (N - 1) * (self.sigma_min**inv_rho - self.sigma_max**inv_rho)
        ) ** self.rho

        sigmas = F.pad(sigmas, (0, 1), value=0.0)
        return sigmas

    @torch.no_grad()
    def sample(
        self,
        self_cond: torch.Tensor,
        batch_size: Optional[int] = None,
        num_sample_steps: Optional[int] = None,
        clamp: bool = True,
        apply_projection: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples using the standard sampling procedure.

        Args:
            self_cond: Self-conditioning input
            batch_size: Number of samples to generate (default: derived from self_cond)
            num_sample_steps: Number of sampling steps (default: self.num_sample_steps)
            clamp: Whether to clamp final outputs to [-1, 1]

        Returns:
            Generated image samples
        """
        batch_size = self_cond.shape[0]
        num_sample_steps = default(num_sample_steps, self.num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)

        sigmas = self.sample_schedule(num_sample_steps)

        gammas = torch.where(
            (sigmas >= self.S_tmin) & (sigmas <= self.S_tmax),
            min(self.S_churn / num_sample_steps, sqrt(2) - 1),
            0.0,
        )

        sigmas_and_gammas = list(zip(sigmas[:-1], sigmas[1:], gammas[:-1]))

        init_sigma = sigmas[0]

        images = init_sigma * torch.randn(shape, device=self.device)

        for sigma, sigma_next, gamma in tqdm(
            sigmas_and_gammas, desc="sampling time step"
        ):
            sigma, sigma_next, gamma = map(
                lambda t: t.item(), (sigma, sigma_next, gamma)
            )

            eps = self.S_noise * torch.randn(shape, device=self.device)

            sigma_hat = sigma + gamma * sigma
            images_hat = images + sqrt(sigma_hat**2 - sigma**2) * eps

            model_output = self.preconditioned_network_forward(
                images_hat, sigma_hat, self_cond, clamp=clamp
            )

            denoised_over_sigma = (images_hat - model_output) / sigma_hat

            images_next = images_hat + (sigma_next - sigma_hat) * denoised_over_sigma

            if sigma_next != 0:
                model_output_next = self.preconditioned_network_forward(
                    images_next, sigma_next, self_cond, clamp=clamp
                )

                denoised_prime_over_sigma = (
                    images_next - model_output_next
                ) / sigma_next
                images_next = images_hat + 0.5 * (sigma_next - sigma_hat) * (
                    denoised_over_sigma + denoised_prime_over_sigma
                )

            images = images_next

        images = images.clamp(-1.0, 1.0)
        if (
            self.helmholtz_projection
            and apply_projection
            and self.projection_mode != "full_field_residual"
        ):
            images = self.projection_module(images)
        return images

    @torch.no_grad()
    def sample_using_dpmpp(
        self,
        self_cond: torch.Tensor,
        batch_size: Optional[int] = None,
        num_sample_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate samples using the DPM++ solver.

        Implements the DPM++ solver from "DPM-Solver: A Fast ODE Solver for Diffusion
        Probabilistic Model Sampling in Around 10 Steps" for improved sampling quality.

        Thanks to Katherine Crowson (https://github.com/crowsonkb) for figuring it all out!
        https://arxiv.org/abs/2211.01095

        Args:
            self_cond: Self-conditioning input
            batch_size: Number of samples to generate (default: derived from self_cond)
            num_sample_steps: Number of sampling steps (default: self.num_sample_steps)

        Returns:
            Generated image samples
        """
        batch_size = self_cond.shape[0]
        device, num_sample_steps = (
            self.device,
            default(num_sample_steps, self.num_sample_steps),
        )

        sigmas = self.sample_schedule(num_sample_steps)

        shape = (batch_size, self.channels, self.image_size, self.image_size)
        images = sigmas[0] * torch.randn(shape, device=device)

        sigma_fn = lambda t: t.neg().exp()
        t_fn = lambda sigma: sigma.log().neg()

        old_denoised = None
        for i in tqdm(range(len(sigmas) - 1)):
            denoised = self.preconditioned_network_forward(
                images, sigmas[i].item(), self_cond
            )

            t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
            h = t_next - t

            if not exists(old_denoised) or sigmas[i + 1] == 0:
                denoised_d = denoised
            else:
                h_last = t - t_fn(sigmas[i - 1])
                r = h_last / h
                gamma = -1 / (2 * r)
                denoised_d = (1 - gamma) * denoised + gamma * old_denoised

            images = (sigma_fn(t_next) / sigma_fn(t)) * images - (
                -h
            ).expm1() * denoised_d
            old_denoised = denoised

        images = images.clamp(-1.0, 1.0)
        if (
            self.helmholtz_projection
            and self.projection_mode != "full_field_residual"
        ):
            images = self.projection_module(images)
        return images

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Calculate the weighting factor for the loss function.

        This implements the λ(σ) weighting from the paper.

        Args:
            sigma: Noise level

        Returns:
            Loss weighting factor
        """
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def noise_distribution(self, batch_size: int) -> torch.Tensor:
        """
        Sample noise levels from the training noise distribution.

        Samples from a log-normal distribution as described in the paper.

        Args:
            batch_size: Number of noise levels to sample

        Returns:
            Batch of noise level values
        """
        return (
            self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)
        ).exp()

    def curl_2d(self, qx: torch.Tensor, qy: torch.Tensor) -> torch.Tensor:
        """
        Calculate the z-curl of a 2D vector field with spectral derivatives.

        Args:
            qx: x-component with shape [batch, time_or_channel, nx, ny]
            qy: y-component with shape [batch, time_or_channel, nx, ny]

        Returns:
            d(qy)/dx - d(qx)/dy with shape [batch, time_or_channel, nx, ny]
        """
        device = qx.device
        dtype = qx.dtype
        nx = qx.size(2)
        ny = qx.size(3)
        kx = (
            2
            * np.pi
            / self.domain_size_x
            * torch.cat(
                [
                    torch.arange(0, nx // 2, device=device),
                    torch.arange(-nx // 2, 0, device=device),
                ]
            )
            .reshape(1, 1, nx, 1)
            .repeat(1, 1, 1, ny)
        )
        ky = (
            2
            * np.pi
            / self.domain_size_y
            * torch.cat(
                [
                    torch.arange(0, ny // 2, device=device),
                    torch.arange(-ny // 2, 0, device=device),
                ]
            )
            .reshape(1, 1, 1, ny)
            .repeat(1, 1, nx, 1)
        )

        kx = kx.to(dtype)
        ky = ky.to(dtype)

        qx_h = torch.fft.fftn(qx, dim=[2, 3])
        qy_h = torch.fft.fftn(qy, dim=[2, 3])
        curl_h = 1j * kx * qy_h - 1j * ky * qx_h
        curl = torch.fft.ifftn(curl_h, dim=[2, 3]).real
        return curl

    def derivative_losses(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute optional relative MSE losses for vorticity and current."""
        loss = pred.new_tensor(0.0)

        eps = 1e-12

        if self.use_vorticity_loss:
            omega_pred = self.curl_2d(pred[:, 0:1], pred[:, 1:2])
            omega_target = self.curl_2d(target[:, 0:1], target[:, 1:2])
            omega_mse = F.mse_loss(omega_pred, omega_target, reduction="none")
            omega_mse = reduce(omega_mse, "b ... -> b", "mean")
            omega_power = reduce(omega_target.square(), "b ... -> b", "mean")
            omega_loss = omega_mse / (omega_power + eps)
            loss = loss + self.vorticity_loss_weight * omega_loss

        if self.use_current_loss:
            current_pred = self.curl_2d(pred[:, 2:3], pred[:, 3:4])
            current_target = self.curl_2d(target[:, 2:3], target[:, 3:4])
            current_mse = F.mse_loss(current_pred, current_target, reduction="none")
            current_mse = reduce(current_mse, "b ... -> b", "mean")
            current_power = reduce(current_target.square(), "b ... -> b", "mean")
            current_loss = current_mse / (current_power + eps)
            loss = loss + self.current_loss_weight * current_loss

        return loss

    def forward(
        self,
        images: torch.Tensor,
        self_cond: Optional[torch.Tensor] = None,
        re: Optional[torch.Tensor] = None,
        input_normalizer=None,
        target_normalizer=None,
        channel_indices=None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            images: Clean input images
            self_cond: Optional self-conditioning input

        Returns:
            Mean loss value

        Raises:
            AssertionError: If image dimensions don't match expected size
        """
        batch_size, c, h, w, device, image_size, channels = (
            *images.shape,
            images.device,
            self.image_size,
            self.channels,
        )

        assert (
            h == image_size and w == image_size
        ), f"height and width of image must be {image_size}"
        assert c == channels, "mismatch of image channels"

        images = images

        sigmas = self.noise_distribution(batch_size)
        padded_sigmas = rearrange(sigmas, "b -> b 1 1 1")

        noise = torch.randn_like(images)

        noised_images = images + padded_sigmas * noise

        denoised = self.preconditioned_network_forward(
            noised_images, sigmas, self_cond
        )
        if self.helmholtz_projection:
            if self.projection_mode == "full_field_residual":
                if self_cond is None:
                    raise ValueError(
                        "full_field_residual projection requires self_cond."
                    )
                denoised = project_residual_via_full_field(
                    self_cond,
                    denoised,
                    input_normalizer,
                    target_normalizer,
                    channel_indices,
                    self.projection_module,
                    re=re,
                )
            else:
                denoised = self.projection_module(denoised)

        losses = F.mse_loss(denoised, images, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        if self.use_vorticity_loss or self.use_current_loss:
            losses = losses + self.derivative_losses(denoised, images)

        losses = losses * self.loss_weight(sigmas)

        return losses.mean()


@register_diffusion_model(model_type="elucidated")
def elucidated_diffusion_default(model_params: Dict[str, Any]) -> ElucidatedDiffusion:
    """
    Create a default ElucidatedDiffusion model with UNet backbone.

    This function constructs a UNet backbone with appropriate parameters
    from the model_params dictionary and wraps it in an ElucidatedDiffusion model.

    Args:
        model_params: Dictionary of model parameters

    Returns:
        Initialized ElucidatedDiffusion model
    """

    unet = UNet(
        dim=model_params.get("base_dim", 64),
        dim_mults=model_params.get("dim_mults", (1, 2, 4, 8)),
        channels=model_params.get("channels", 3),
        self_condition=model_params.get("self_condition", True),
        learned_variance=model_params.get("learned_variance", False),
        learned_sinusoidal_cond=model_params.get("learned_sinusoidal_cond", False),
        random_fourier_features=model_params.get("random_fourier_features", False),
        flash_attn=model_params.get("flash_attn", False),
        attn_heads=model_params.get("attn_heads", 4),
        attn_dim_head=model_params.get("attn_dim_head", 32),
        dropout=model_params.get("dropout", 0.0),
        padding_mode=model_params.get("padding_mode", "zeros"),
    )

    model = ElucidatedDiffusion(
        unet,
        image_size=model_params.get("image_size", 128),
        channels=model_params.get("channels", 3),
        num_sample_steps=model_params.get("num_sample_steps", 32),
        sigma_min=model_params.get("sigma_min", 0.002),
        sigma_max=model_params.get("sigma_max", 80),
        sigma_data=model_params.get("sigma_data", 0.5),
        rho=model_params.get("rho", 7),
        P_mean=model_params.get("P_mean", -1.2),
        P_std=model_params.get("P_std", 1.2),
        S_churn=model_params.get("S_churn", 80),
        S_tmin=model_params.get("S_tmin", 0.05),
        S_tmax=model_params.get("S_tmax", 50),
        S_noise=model_params.get("S_noise", 1.003),
        helmholtz_projection=model_params.get("helmholtz_projection", False),
        project_velocity=model_params.get("project_velocity", True),
        project_B=model_params.get("project_B", True),
        projection_mode=model_params.get("projection_mode", "output"),
        domain_size_x=model_params.get("domain_size_x", 1.0),
        domain_size_y=model_params.get("domain_size_y", 1.0),
        use_vorticity_loss=model_params.get("use_vorticity_loss", False),
        use_current_loss=model_params.get("use_current_loss", False),
        vorticity_loss_weight=model_params.get("vorticity_loss_weight", 1.0),
        current_loss_weight=model_params.get("current_loss_weight", 5.0),
    )

    return model
