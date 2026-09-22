### Adapted for DINOs diffusion model by adapting mhd_pino_repo version
### To enforce divergence free conditions on the velocity and magnetic field
### use Helmholtz projection to remove compressible modes
### implementation based on Li et al. 2026 https://arxiv.org/abs/2603.24500

import torch
import torch.nn as nn


class HelmholtzProjectionDiffusion(nn.Module):
    """
    Helmholtz projection for DINOs diffusion model output.

    Projects velocity (u, v) and magnetic field (Bx, By) components
    to their divergence-free (solenoidal) components using Helmholtz
    decomposition in Fourier space.

    Input tensor format: (batch, channels, height, width)
    Supports 3, 4, or 6 channels.
    """

    def __init__(self, Lx=1.0, Ly=1.0, project_velocity=True, project_B=True):
        super().__init__()
        self.Lx = Lx
        self.Ly = Ly
        self.project_velocity = project_velocity
        self.project_B = project_B

    # Helmholtz decomposition to remove compressible modes
    def project_div_free(self, qx, qy):
        """
        Project 2D vector field (qx, qy) to divergence-free component.

        Args:
            qx: x-component, shape (batch, height, width)
            qy: y-component, shape (batch, height, width)

        Returns:
            (qx_proj, qy_proj): Divergence-free components
        """
        orig_dtype = qx.dtype
        qx = qx.double()
        qy = qy.double()

        device = qx.device
        nx = qx.size(-2)  # height dimension
        ny = qx.size(-1)   # width dimension

        kx = 2.0 * torch.pi * torch.fft.fftfreq(
            nx, d=self.Lx / nx, device=device, dtype=qx.dtype
        ).reshape(1, nx, 1)
        ky = 2.0 * torch.pi * torch.fft.fftfreq(
            ny, d=self.Ly / ny, device=device, dtype=qx.dtype
        ).reshape(1, 1, ny)

        qx_h = torch.fft.fft2(qx, dim=[-2, -1])
        qy_h = torch.fft.fft2(qy, dim=[-2, -1])

        k2 = kx.square() + ky.square()
        k2[..., 0, 0] = 1.0

        k_dot_q = kx * qx_h + ky * qy_h
        qx_h_proj = qx_h - kx * k_dot_q / k2
        qy_h_proj = qy_h - ky * k_dot_q / k2

        qx_h_proj[..., 0, 0] = 0.0
        qy_h_proj[..., 0, 0] = 0.0

        # Even grids have Nyquist rows/columns that cannot carry arbitrary
        # real-valued divergence-free vector modes. Zero them to avoid the
        # residual compressible energy that survives after taking ifft(...).real.
        if nx % 2 == 0:
            qx_h_proj[:, nx // 2, :] = 0.0
            qy_h_proj[:, nx // 2, :] = 0.0
        if ny % 2 == 0:
            qx_h_proj[:, :, ny // 2] = 0.0
            qy_h_proj[:, :, ny // 2] = 0.0

        qx_proj = torch.fft.ifft2(qx_h_proj, dim=[-2, -1]).real
        qy_proj = torch.fft.ifft2(qy_h_proj, dim=[-2, -1]).real

        return qx_proj.to(orig_dtype), qy_proj.to(orig_dtype)

    def forward(self, pred):
        """
        Apply Helmholtz projection to diffusion model output.

        Input format: (batch, channels, height, width)
        Supported channel configurations:
        - 3 channels: [ux, uy, A] where A is magnetic vector potential (already divergence-free)
        - 4 channels: [ux, uy, Bx, By] where Bx, By are magnetic field components
        - 6 channels: [ux, uy, Bx, By, omega, J]

        Args:
            pred: Model output tensor

        Returns:
            pred_proj: Projection-applied output (same shape as pred)
        """
        # Extract velocity components (always present)
        ux = pred[:, 0, :, :]
        uy = pred[:, 1, :, :]

        # Project velocity to be divergence-free
        if self.project_velocity:
            ux, uy = self.project_div_free(ux, uy)

        # 3-channel case: [ux, uy, A]
        # A (vector potential) is already divergence-free by construction, no projection needed
        if pred.shape[1] == 3:
            A = pred[:, 2, :, :]
            pred_proj = torch.stack([ux, uy, A], dim=1)

        # 4-channel case: [ux, uy, Bx, By]
        elif pred.shape[1] == 4:
            Bx = pred[:, 2, :, :]
            By = pred[:, 3, :, :]

            # Project magnetic field if enabled
            if self.project_B:
                Bx, By = self.project_div_free(Bx, By)

            pred_proj = torch.stack([ux, uy, Bx, By], dim=1)

        # 6-channel case: [ux, uy, Bx, By, omega, J]
        elif pred.shape[1] == 6:
            Bx = pred[:, 2, :, :]
            By = pred[:, 3, :, :]
            omega = pred[:, 4, :, :]
            J = pred[:, 5, :, :]

            # Project magnetic field if enabled
            if self.project_B:
                Bx, By = self.project_div_free(Bx, By)

            pred_proj = torch.stack([ux, uy, Bx, By, omega, J], dim=1)

        else:
            raise ValueError(f"Unsupported number of channels: {pred.shape[1]}. Expected 3, 4, or 6.")

        return pred_proj
