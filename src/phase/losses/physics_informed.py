"""Physics-informed loss functions that incorporate domain knowledge constraints."""

from typing import Dict, Any, Optional, Tuple, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch import Tensor

from .loss_factory import register_loss
from .lp_loss import LpLoss
from ..physics.constraints import compute_constraints, compute_constraint_loss
from ..physics.pde_solvers import (
    compute_bfield_pde_loss,
    compute_mhd_bfield_pde,
    compute_mhd_pde,
    compute_pde_loss,
)
from ..utils.fourier_utils import create_wavenumbers, compute_derivative

from ..utils import (
    get_dataset_normalizer,
    identify_data_channels,
    apply_denormalization,
)


class MHDVecPotLoss(nn.Module):
    """
    Physics-informed loss function for MHD with vector potential formulation.

    This loss combines data fitting with physical constraints from the MHD equations:
    - Data fitting against ground truth predictions
    - Initial condition matching
    - PDE residual minimization
    - Divergence-free constraints for velocity and magnetic fields

    Attributes:
        requires_input_data: Flag indicating this loss requires input data
    """

    # Class flag to indicate this loss requires input data
    requires_input_data: bool = True

    def __init__(
        self,
        nu: float = 1e-4,
        eta: float = 1e-4,
        rho0: float = 1.0,
        data_weight: float = 1.0,
        ic_weight: float = 1.0,
        pde_weight: float = 1.0,
        constraint_weight: float = 1.0,
        use_data_loss: bool = True,
        use_ic_loss: bool = True,
        use_pde_loss: bool = True,
        use_constraint_loss: bool = True,
        u_weight: float = 1.0,
        v_weight: float = 1.0,
        A_weight: float = 1.0,
        Du_weight: float = 1.0,
        Dv_weight: float = 1.0,
        DA_weight: float = 1.0,
        div_B_weight: float = 1.0,
        div_vel_weight: float = 1.0,
        magnetic_field_weight: float = 0.0,
        log_inactive_derived_components: bool = False,
        Lx: float = 1.0,
        Ly: float = 1.0,
        tend: float = 1.0,
        use_weighted_mean: bool = False,
        **kwargs,  # Accept and ignore additional keyword arguments
    ):
        """
        Initialize the MHD vector potential loss function.

        Args:
            nu: Kinematic viscosity
            eta: Magnetic diffusivity
            rho0: Fluid density
            data_weight: Weight for data fitting loss
            ic_weight: Weight for initial condition loss
            pde_weight: Weight for PDE residual loss
            constraint_weight: Weight for constraint loss
            use_data_loss: Whether to include data fitting loss
            use_ic_loss: Whether to include initial condition loss
            use_pde_loss: Whether to include PDE residual loss
            use_constraint_loss: Whether to include constraint loss
            u_weight: Weight for x-velocity component
            v_weight: Weight for y-velocity component
            A_weight: Weight for vector potential
            Du_weight: Weight for u-momentum equation residual
            Dv_weight: Weight for v-momentum equation residual
            DA_weight: Weight for vector potential equation residual
            div_B_weight: Weight for magnetic field divergence constraint
            div_vel_weight: Weight for velocity divergence constraint
            log_inactive_derived_components: Record zero-valued vorticity and
                current entries for compatibility with later legacy logs
            Lx: Domain length in x direction
            Ly: Domain length in y direction
            tend: Final simulation time
            use_weighted_mean: Whether to normalize by sum of weights
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__()
        self.nu = nu
        self.eta = eta
        self.rho0 = rho0

        # Main loss component weights
        self.data_weight = data_weight
        self.ic_weight = ic_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight

        # Loss component flags
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.use_constraint_loss = use_constraint_loss

        # Field component weights
        self.u_weight = u_weight
        self.v_weight = v_weight
        self.A_weight = A_weight

        # PDE residual weights
        self.Du_weight = Du_weight
        self.Dv_weight = Dv_weight
        self.DA_weight = DA_weight

        # Constraint weights
        self.div_B_weight = div_B_weight
        self.div_vel_weight = div_vel_weight
        self.magnetic_field_weight = magnetic_field_weight
        self.log_inactive_derived_components = log_inactive_derived_components

        # Domain parameters
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend

        # Loss calculation method
        self.use_weighted_mean = use_weighted_mean
        self.last_components = {}

    def forward(
        self,
        dataloader: DataLoader,
        pred: Tensor,
        target: Tensor,
        inputs: Tensor,
        metadata: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Calculate the total loss combining data fit and physics constraints.

        Args:
            dataloader: DataLoader containing the dataset (for normalizer access)
            pred: Model predictions with shape [batch, channels, time, x, y]
            target: Ground truth targets with shape [batch, channels, time, x, y]
            inputs: Input data containing initial conditions

        Returns:
            Total weighted loss
        """
        normalizer = get_dataset_normalizer(dataloader)

        # Identify data channels and check for grid embeddings
        data_channel_indices, has_grid_embeddings, _ = identify_data_channels(
            inputs, target
        )

        # Denormalize the data if a normalizer is available
        inputs_denorm, target_denorm, pred_denorm = apply_denormalization(
            inputs, target, pred, normalizer, data_channel_indices, has_grid_embeddings
        )

        return self.compute_loss(pred_denorm, target_denorm, inputs_denorm, metadata)

    def _transport_coefficients(
        self, metadata: Optional[Dict[str, Tensor]], pred: Tensor
    ) -> Tuple[Union[float, Tensor], Union[float, Tensor]]:
        if metadata is None:
            return self.nu, self.eta

        nu = metadata.get("nu")
        eta = metadata.get("eta")
        if nu is None and metadata.get("re") is not None:
            nu = 1.0 / metadata["re"].to(pred.device, dtype=pred.dtype)
        if eta is None:
            if metadata.get("rem") is not None:
                eta = 1.0 / metadata["rem"].to(pred.device, dtype=pred.dtype)
            elif metadata.get("re") is not None:
                eta = 1.0 / metadata["re"].to(pred.device, dtype=pred.dtype)

        if nu is None:
            nu = self.nu
        if eta is None:
            eta = self.eta

        if torch.is_tensor(nu):
            nu = nu.to(pred.device, dtype=pred.dtype).view(-1, 1, 1, 1)
        if torch.is_tensor(eta):
            eta = eta.to(pred.device, dtype=pred.dtype).view(-1, 1, 1, 1)
        return nu, eta

    def compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        inputs: Tensor,
        metadata: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """
        Compute weighted loss with all components.

        Args:
            pred: Predicted fields [batch, channels, time, x, y]
            target: Target fields [batch, channels, time, x, y]
            inputs: Input fields [batch, channels, time, x, y]

        Returns:
            Total weighted loss
        """
        # Extract field components
        u, v, A = pred[:, 0], pred[:, 1], pred[:, 2]

        # Track individual loss components
        loss_components = {}

        # Data loss
        if self.use_data_loss:
            loss_data, data_components = self.data_loss(
                pred, target, return_components=True
            )
            loss_components["data"] = loss_data.item()
            loss_components.update({f"data_{k}": v for k, v in data_components.items()})
        else:
            loss_data = torch.tensor(0.0, device=pred.device)
            loss_components["data"] = 0.0

        # Initial condition loss
        if self.use_ic_loss:
            loss_ic, ic_components = self.ic_loss(
                pred, inputs, return_components=True
            )
            loss_components["ic"] = loss_ic.item()
            loss_components.update({f"ic_{k}": v for k, v in ic_components.items()})
        else:
            loss_ic = torch.tensor(0.0, device=pred.device)
            loss_components["ic"] = 0.0

        # PDE loss
        if self.use_pde_loss:
            nu, eta = self._transport_coefficients(metadata, pred)
            Du, Dv, DA = compute_mhd_pde(
                u, v, A, self.Lx, self.Ly, self.tend, nu, eta, self.rho0
            )
            loss_pde, pde_components = compute_pde_loss(
                Du,
                Dv,
                DA,
                self.Du_weight,
                self.Dv_weight,
                self.DA_weight,
                self.use_weighted_mean,
            )
            loss_components.update({f"pde_{k}": v for k, v in pde_components.items()})
        else:
            loss_pde = torch.tensor(0.0, device=pred.device)

        # Constraint loss
        if self.use_constraint_loss:
            div_vel, div_B = compute_constraints(u, v, A, self.Lx, self.Ly, self.tend)
            loss_constraint, constraint_components = compute_constraint_loss(
                div_vel,
                div_B,
                self.div_vel_weight,
                self.div_B_weight,
                self.use_weighted_mean,
            )
            loss_components.update(
                {f"constraint_{k}": v for k, v in constraint_components.items()}
            )
        else:
            loss_constraint = torch.tensor(0.0, device=pred.device)

        if self.magnetic_field_weight > 0:
            loss_magnetic_field, magnetic_field_components = self.magnetic_field_loss(
                pred[:, 2], target[:, 2], return_components=True
            )
            loss_components["magnetic_field"] = loss_magnetic_field.item()
            loss_components.update(
                {f"magnetic_field_{k}": v for k, v in magnetic_field_components.items()}
            )
        else:
            loss_magnetic_field = torch.tensor(0.0, device=pred.device)
            loss_components["magnetic_field"] = 0.0

        if self.log_inactive_derived_components:
            loss_components.update({"vorticity": 0.0, "current": 0.0})

        # Calculate weight normalization factor
        if self.use_weighted_mean:
            active_weights = (
                (self.data_weight if self.use_data_loss else 0)
                + (self.ic_weight if self.use_ic_loss else 0)
                + (self.pde_weight if self.use_pde_loss else 0)
                + (self.constraint_weight if self.use_constraint_loss else 0)
                + self.magnetic_field_weight
            )
            weight_sum = max(active_weights, 1.0)  # Avoid division by zero
        else:
            weight_sum = 1.0

        # Compute weighted total loss
        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
            + self.constraint_weight * loss_constraint
            + self.magnetic_field_weight * loss_magnetic_field
        ) / weight_sum

        # Store total loss
        loss_components["total"] = loss.item()
        self.last_components = loss_components

        return loss

    def magnetic_field_loss(
        self, A_pred: Tensor, A_target: Tensor, return_components: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        """Relative L2 loss on B = curl(A), computed with spectral derivatives."""
        lploss = LpLoss(size_average=True)
        Bx_pred, By_pred = self.vector_potential_to_B(A_pred)
        Bx_target, By_target = self.vector_potential_to_B(A_target)
        loss_Bx = lploss(Bx_pred, Bx_target)
        loss_By = lploss(By_pred, By_target)
        loss_B = 0.5 * (loss_Bx + loss_By)
        if return_components:
            return loss_B, {"Bx": loss_Bx.item(), "By": loss_By.item()}
        return loss_B

    def vector_potential_to_B(self, A: Tensor) -> Tuple[Tensor, Tensor]:
        nx = A.size(2)
        ny = A.size(3)
        k_x, k_y = create_wavenumbers(nx, ny, self.Lx, self.Ly, A.device)
        A_h = torch.fft.fftn(A, dim=[2, 3])
        Ax_h = compute_derivative(A_h, k_x)
        Ay_h = compute_derivative(A_h, k_y)
        Bx = torch.fft.ifftn(Ay_h, dim=[2, 3]).real
        By = torch.fft.ifftn(-Ax_h, dim=[2, 3]).real
        return Bx, By

    def data_loss(
        self, pred: Tensor, target: Tensor, return_components: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        """
        Compute data fitting loss using Lp loss.

        Args:
            pred: Predicted fields [batch, channels, time, x, y]
            target: Target fields [batch, channels, time, x, y]

        Returns:
            Data fitting loss
        """
        lploss = LpLoss(size_average=True)

        # Extract field components
        u_pred, v_pred, A_pred = pred[:, 0], pred[:, 1], pred[:, 2]
        u_target, v_target, A_target = target[:, 0], target[:, 1], target[:, 2]

        # Compute component losses
        loss_u = lploss(u_pred, u_target)
        loss_v = lploss(v_pred, v_target)
        loss_A = lploss(A_pred, A_target)

        # Apply component weights
        if self.use_weighted_mean:
            weight_sum = self.u_weight + self.v_weight + self.A_weight
        else:
            weight_sum = 1.0

        # Compute weighted loss
        loss_data = (
            self.u_weight * loss_u + self.v_weight * loss_v + self.A_weight * loss_A
        ) / weight_sum

        if return_components:
            return loss_data, {
                "u": loss_u.item(),
                "v": loss_v.item(),
                "A": loss_A.item(),
            }
        return loss_data

    def ic_loss(
        self, pred: Tensor, inputs: Tensor, return_components: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        """
        Compute initial condition loss using Lp loss.

        Args:
            pred: Predicted fields [batch, channels, time, x, y]
            inputs: Input fields with initial conditions [batch, channels, time, x, y]

        Returns:
            Initial condition loss
        """
        lploss = LpLoss(size_average=True)

        # Extract initial conditions (t=0)
        ic_pred = pred[:, :, 0]
        ic_target = inputs[:, :, 0]

        # Extract field components at t=0
        u_ic_pred, v_ic_pred, A_ic_pred = ic_pred[:, 0], ic_pred[:, 1], ic_pred[:, 2]
        u_ic_target, v_ic_target, A_ic_target = (
            ic_target[:, 0],
            ic_target[:, 1],
            ic_target[:, 2],
        )

        # Compute component losses
        loss_u_ic = lploss(u_ic_pred, u_ic_target)
        loss_v_ic = lploss(v_ic_pred, v_ic_target)
        loss_A_ic = lploss(A_ic_pred, A_ic_target)

        # Apply component weights
        if self.use_weighted_mean:
            weight_sum = self.u_weight + self.v_weight + self.A_weight
        else:
            weight_sum = 1.0

        # Compute weighted loss
        loss_ic = (
            self.u_weight * loss_u_ic
            + self.v_weight * loss_v_ic
            + self.A_weight * loss_A_ic
        ) / weight_sum

        if return_components:
            return loss_ic, {
                "u": loss_u_ic.item(),
                "v": loss_v_ic.item(),
                "A": loss_A_ic.item(),
            }
        return loss_ic


@register_loss("physics-informed")
def create_physics_informed_loss(params: Dict[str, Any]) -> MHDVecPotLoss:
    """
    Create a physics-informed loss function for MHD with vector potential formulation.

    Args:
        params: Dictionary of parameters for the loss function

    Returns:
        Configured MHDVecPotLoss instance
    """
    # You can optionally filter out the 'type' key here if you want
    # params_copy = {k: v for k, v in params.items() if k != 'type'}
    # return MHDVecPotLoss(**params_copy)

    # Or simply pass all params which is fine since we added **kwargs to the constructor
    return MHDVecPotLoss(**params)

class MHDDirectBFieldLoss(nn.Module):
    """Physics-aware loss for direct magnetic-field representation [u, v, Bx, By]."""

    requires_input_data: bool = True

    def __init__(
        self,
        nu: float = 1e-4,
        eta: float = 1e-4,
        rho0: float = 1.0,
        data_weight: float = 1.0,
        ic_weight: float = 1.0,
        pde_weight: float = 1.0,
        constraint_weight: float = 1.0,
        vorticity_weight: float = 0.0,
        current_weight: float = 0.0,
        delta_B_weight: float = 0.0,
        use_data_loss: bool = True,
        use_ic_loss: bool = True,
        use_pde_loss: bool = False,
        use_constraint_loss: bool = True,
        u_weight: float = 1.0,
        v_weight: float = 1.0,
        Bx_weight: float = 1.0,
        By_weight: float = 1.0,
        Du_weight: float = 1.0,
        Dv_weight: float = 1.0,
        DBx_weight: float = 1.0,
        DBy_weight: float = 1.0,
        delta_Bx_weight: float = 1.0,
        delta_By_weight: float = 1.0,
        div_vel_weight: float = 1.0,
        div_B_weight: float = 1.0,
        Lx: float = 1.0,
        Ly: float = 1.0,
        tend: float = 1.0,
        use_weighted_mean: bool = False,
        u_time_loss_mode: str = "global",
        u_time_loss_eps: float = 1e-6,
        v_time_loss_mode: str = "global",
        v_time_loss_eps: float = 1e-6,
        magnetic_time_loss_mode: str = "global",
        magnetic_time_loss_eps: float = 1e-6,
        derived_time_loss_mode: str = "global",
        derived_time_loss_eps: float = 1e-6,
        vorticity_time_loss_mode: Optional[str] = None,
        vorticity_time_loss_eps: Optional[float] = None,
        current_time_loss_mode: Optional[str] = None,
        current_time_loss_eps: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.nu = nu
        self.eta = eta
        self.rho0 = rho0
        self.data_weight = data_weight
        self.ic_weight = ic_weight
        self.pde_weight = pde_weight
        self.constraint_weight = constraint_weight
        self.vorticity_weight = vorticity_weight
        self.current_weight = current_weight
        self.delta_B_weight = delta_B_weight
        self.use_data_loss = use_data_loss
        self.use_ic_loss = use_ic_loss
        self.use_pde_loss = use_pde_loss
        self.use_constraint_loss = use_constraint_loss
        self.u_weight = u_weight
        self.v_weight = v_weight
        self.Bx_weight = Bx_weight
        self.By_weight = By_weight
        self.Du_weight = Du_weight
        self.Dv_weight = Dv_weight
        self.DBx_weight = DBx_weight
        self.DBy_weight = DBy_weight
        self.delta_Bx_weight = delta_Bx_weight
        self.delta_By_weight = delta_By_weight
        self.div_vel_weight = div_vel_weight
        self.div_B_weight = div_B_weight
        self.Lx = Lx
        self.Ly = Ly
        self.tend = tend
        self.use_weighted_mean = use_weighted_mean
        self.u_time_loss_mode = self._validate_time_loss_mode(
            "u_time_loss_mode", u_time_loss_mode
        )
        self.u_time_loss_eps = u_time_loss_eps
        self.v_time_loss_mode = self._validate_time_loss_mode(
            "v_time_loss_mode", v_time_loss_mode
        )
        self.v_time_loss_eps = v_time_loss_eps
        self.magnetic_time_loss_mode = self._validate_time_loss_mode(
            "magnetic_time_loss_mode", magnetic_time_loss_mode
        )
        self.magnetic_time_loss_eps = magnetic_time_loss_eps
        self.derived_time_loss_mode = self._validate_time_loss_mode(
            "derived_time_loss_mode", derived_time_loss_mode
        )
        self.derived_time_loss_eps = derived_time_loss_eps
        self.vorticity_time_loss_mode = self._validate_time_loss_mode(
            "vorticity_time_loss_mode",
            vorticity_time_loss_mode or self.derived_time_loss_mode,
        )
        self.vorticity_time_loss_eps = (
            derived_time_loss_eps
            if vorticity_time_loss_eps is None
            else vorticity_time_loss_eps
        )
        self.current_time_loss_mode = self._validate_time_loss_mode(
            "current_time_loss_mode",
            current_time_loss_mode or self.derived_time_loss_mode,
        )
        self.current_time_loss_eps = (
            derived_time_loss_eps
            if current_time_loss_eps is None
            else current_time_loss_eps
        )
        self.last_components = {}

    @staticmethod
    def _validate_time_loss_mode(name: str, mode: str) -> str:
        if mode not in {"global", "time_relative"}:
            raise ValueError(
                f"{name} must be global or time_relative, got {mode!r}."
            )
        return mode

    @staticmethod
    def _time_relative_component_loss(
        pred: Tensor, target: Tensor, eps: float
    ) -> Tensor:
        """Average spatial relative L2 independently over sample and time."""
        if pred.ndim != 4 or target.ndim != 4:
            raise ValueError(
                "Time-relative component loss expects tensors shaped [B, T, H, W]."
            )
        error = torch.linalg.vector_norm((pred - target).flatten(2), dim=-1)
        scale = torch.linalg.vector_norm(target.flatten(2), dim=-1)
        return (error / (scale + eps)).mean()

    def _component_loss(
        self, pred: Tensor, target: Tensor, mode: str, eps: float
    ) -> Tensor:
        if mode == "time_relative":
            return self._time_relative_component_loss(pred, target, eps)
        return LpLoss(size_average=True)(pred, target)

    def forward(
        self,
        dataloader: DataLoader,
        pred: Tensor,
        target: Tensor,
        inputs: Tensor,
        metadata: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        normalizer = get_dataset_normalizer(dataloader)
        data_channel_indices, has_grid_embeddings, _ = identify_data_channels(
            inputs, target
        )
        inputs_denorm, target_denorm, pred_denorm = apply_denormalization(
            inputs, target, pred, normalizer, data_channel_indices, has_grid_embeddings
        )
        return self.compute_loss(pred_denorm, target_denorm, inputs_denorm, metadata)

    def _transport_coefficients(
        self, metadata: Optional[Dict[str, Tensor]], pred: Tensor
    ) -> Tuple[Union[float, Tensor], Union[float, Tensor]]:
        if metadata is None:
            return self.nu, self.eta

        nu = metadata.get("nu")
        eta = metadata.get("eta")
        if nu is None and metadata.get("re") is not None:
            nu = 1.0 / metadata["re"].to(pred.device, dtype=pred.dtype)
        if eta is None:
            if metadata.get("rem") is not None:
                eta = 1.0 / metadata["rem"].to(pred.device, dtype=pred.dtype)
            elif metadata.get("re") is not None:
                eta = 1.0 / metadata["re"].to(pred.device, dtype=pred.dtype)

        if nu is None:
            nu = self.nu
        if eta is None:
            eta = self.eta

        if torch.is_tensor(nu):
            nu = nu.to(pred.device, dtype=pred.dtype).view(-1, 1, 1, 1)
        if torch.is_tensor(eta):
            eta = eta.to(pred.device, dtype=pred.dtype).view(-1, 1, 1, 1)
        return nu, eta

    def compute_loss(
        self,
        pred: Tensor,
        target: Tensor,
        inputs: Tensor,
        metadata: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        if pred.size(1) < 4 or target.size(1) < 4 or inputs.size(1) < 4:
            raise ValueError(
                "MHDDirectBFieldLoss expects channel layout [u, v, Bx, By]."
            )

        loss_components = {}
        zero = torch.tensor(0.0, device=pred.device)

        if self.use_data_loss:
            loss_data, data_components = self.data_loss(
                pred, target, return_components=True
            )
            loss_components["data"] = loss_data.item()
            loss_components.update({f"data_{k}": v for k, v in data_components.items()})
        else:
            loss_data = zero
            loss_components["data"] = 0.0

        if self.use_ic_loss:
            loss_ic, ic_components = self.ic_loss(pred, inputs, return_components=True)
            loss_components["ic"] = loss_ic.item()
            loss_components.update({f"ic_{k}": v for k, v in ic_components.items()})
        else:
            loss_ic = zero
            loss_components["ic"] = 0.0

        if self.use_constraint_loss:
            div_vel = self.divergence_2d(pred[:, 0], pred[:, 1])
            div_B = self.divergence_2d(pred[:, 2], pred[:, 3])
            loss_constraint, constraint_components = self.constraint_loss(div_vel, div_B)
            loss_components["constraint"] = loss_constraint.item()
            loss_components.update(
                {f"constraint_{k}": v for k, v in constraint_components.items()}
            )
        else:
            loss_constraint = zero
            loss_components["constraint"] = 0.0

        if self.use_pde_loss:
            nu, eta = self._transport_coefficients(metadata, pred)
            Du, Dv, DBx, DBy = compute_mhd_bfield_pde(
                pred[:, 0],
                pred[:, 1],
                pred[:, 2],
                pred[:, 3],
                self.Lx,
                self.Ly,
                self.tend,
                nu,
                eta,
                self.rho0,
            )
            loss_pde, pde_components = compute_bfield_pde_loss(
                Du,
                Dv,
                DBx,
                DBy,
                self.Du_weight,
                self.Dv_weight,
                self.DBx_weight,
                self.DBy_weight,
                self.use_weighted_mean,
            )
            loss_components["pde"] = loss_pde.item()
            loss_components.update({f"pde_{k}": v for k, v in pde_components.items()})
        else:
            loss_pde = zero
            loss_components["pde"] = 0.0

        if self.vorticity_weight > 0:
            loss_vorticity = self.vorticity_loss(
                pred[:, 0], pred[:, 1], target[:, 0], target[:, 1]
            )
            loss_components["vorticity"] = loss_vorticity.item()
        else:
            loss_vorticity = zero
            loss_components["vorticity"] = 0.0

        if self.current_weight > 0:
            loss_current = self.current_loss(
                pred[:, 2], pred[:, 3], target[:, 2], target[:, 3]
            )
            loss_components["current"] = loss_current.item()
        else:
            loss_current = zero
            loss_components["current"] = 0.0

        if self.delta_B_weight > 0:
            loss_delta_B, delta_B_components = self.delta_B_loss(
                pred, target, inputs, return_components=True
            )
            loss_components["delta_B"] = loss_delta_B.item()
            loss_components.update(
                {f"delta_B_{k}": v for k, v in delta_B_components.items()}
            )
        else:
            loss_delta_B = zero
            loss_components["delta_B"] = 0.0

        if self.use_weighted_mean:
            active_weights = (
                (self.data_weight if self.use_data_loss else 0.0)
                + (self.ic_weight if self.use_ic_loss else 0.0)
                + (self.pde_weight if self.use_pde_loss else 0.0)
                + (self.constraint_weight if self.use_constraint_loss else 0.0)
                + self.vorticity_weight
                + self.current_weight
                + self.delta_B_weight
            )
            weight_sum = max(active_weights, 1.0)
        else:
            weight_sum = 1.0

        loss = (
            self.data_weight * loss_data
            + self.ic_weight * loss_ic
            + self.pde_weight * loss_pde
            + self.constraint_weight * loss_constraint
            + self.vorticity_weight * loss_vorticity
            + self.current_weight * loss_current
            + self.delta_B_weight * loss_delta_B
        ) / weight_sum

        loss_components["total"] = loss.item()
        self.last_components = loss_components
        return loss

    def data_loss(
        self, pred: Tensor, target: Tensor, return_components: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        losses = {
            "u": self._component_loss(
                pred[:, 0], target[:, 0], self.u_time_loss_mode, self.u_time_loss_eps
            ),
            "v": self._component_loss(
                pred[:, 1], target[:, 1], self.v_time_loss_mode, self.v_time_loss_eps
            ),
            "Bx": self._component_loss(
                pred[:, 2], target[:, 2], self.magnetic_time_loss_mode,
                self.magnetic_time_loss_eps
            ),
            "By": self._component_loss(
                pred[:, 3], target[:, 3], self.magnetic_time_loss_mode,
                self.magnetic_time_loss_eps
            ),
        }
        weight_sum = (
            self.u_weight + self.v_weight + self.Bx_weight + self.By_weight
            if self.use_weighted_mean
            else 1.0
        )
        loss_data = (
            self.u_weight * losses["u"]
            + self.v_weight * losses["v"]
            + self.Bx_weight * losses["Bx"]
            + self.By_weight * losses["By"]
        ) / weight_sum
        if return_components:
            return loss_data, {k: v.item() for k, v in losses.items()}
        return loss_data

    def ic_loss(
        self, pred: Tensor, inputs: Tensor, return_components: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        lploss = LpLoss(size_average=True)
        ic_pred = pred[:, :, 0]
        ic_target = inputs[:, :, 0]
        losses = {
            "u": lploss(ic_pred[:, 0], ic_target[:, 0]),
            "v": lploss(ic_pred[:, 1], ic_target[:, 1]),
            "Bx": lploss(ic_pred[:, 2], ic_target[:, 2]),
            "By": lploss(ic_pred[:, 3], ic_target[:, 3]),
        }
        weight_sum = (
            self.u_weight + self.v_weight + self.Bx_weight + self.By_weight
            if self.use_weighted_mean
            else 1.0
        )
        loss_ic = (
            self.u_weight * losses["u"]
            + self.v_weight * losses["v"]
            + self.Bx_weight * losses["Bx"]
            + self.By_weight * losses["By"]
        ) / weight_sum
        if return_components:
            return loss_ic, {k: v.item() for k, v in losses.items()}
        return loss_ic

    def constraint_loss(
        self, div_vel: Tensor, div_B: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        loss_div_vel = F.mse_loss(div_vel, torch.zeros_like(div_vel))
        loss_div_B = F.mse_loss(div_B, torch.zeros_like(div_B))
        weight_sum = (
            self.div_vel_weight + self.div_B_weight
            if self.use_weighted_mean
            else 1.0
        )
        loss_constraint = (
            self.div_vel_weight * loss_div_vel + self.div_B_weight * loss_div_B
        ) / weight_sum
        return loss_constraint, {
            "div_vel": loss_div_vel.item(),
            "div_B": loss_div_B.item(),
        }

    def _wavenumbers(self, q: Tensor) -> Tuple[Tensor, Tensor]:
        nx = q.size(2)
        ny = q.size(3)
        return create_wavenumbers(nx, ny, self.Lx, self.Ly, q.device)

    def divergence_2d(self, qx: Tensor, qy: Tensor) -> Tensor:
        k_x, k_y = self._wavenumbers(qx)
        qx_h = torch.fft.fftn(qx, dim=[2, 3])
        qy_h = torch.fft.fftn(qy, dim=[2, 3])
        div_h = compute_derivative(qx_h, k_x) + compute_derivative(qy_h, k_y)
        return torch.fft.ifftn(div_h, dim=[2, 3]).real

    def curl_2d(self, qx: Tensor, qy: Tensor) -> Tensor:
        k_x, k_y = self._wavenumbers(qx)
        qx_h = torch.fft.fftn(qx, dim=[2, 3])
        qy_h = torch.fft.fftn(qy, dim=[2, 3])
        curl_h = compute_derivative(qy_h, k_x) - compute_derivative(qx_h, k_y)
        return torch.fft.ifftn(curl_h, dim=[2, 3]).real

    def vorticity_loss(
        self,
        u_pred: Tensor,
        v_pred: Tensor,
        u_target: Tensor,
        v_target: Tensor,
    ) -> Tensor:
        omega_pred = self.curl_2d(u_pred, v_pred)
        omega_target = self.curl_2d(u_target, v_target)
        return self._component_loss(
            omega_pred, omega_target, self.vorticity_time_loss_mode,
            self.vorticity_time_loss_eps
        )

    def current_loss(
        self,
        Bx_pred: Tensor,
        By_pred: Tensor,
        Bx_target: Tensor,
        By_target: Tensor,
    ) -> Tensor:
        j_pred = self.curl_2d(Bx_pred, By_pred)
        j_target = self.curl_2d(Bx_target, By_target)
        return self._component_loss(
            j_pred, j_target, self.current_time_loss_mode,
            self.current_time_loss_eps
        )

    def delta_B_loss(
        self,
        pred: Tensor,
        target: Tensor,
        inputs: Tensor,
        return_components: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, float]]]:
        """Relative L2 loss on magnetic increments B(t) - B(0)."""
        lploss = LpLoss(size_average=True)
        B0x = inputs[:, 2, 0].unsqueeze(1)
        B0y = inputs[:, 3, 0].unsqueeze(1)
        losses = {
            "Bx": lploss(pred[:, 2] - B0x, target[:, 2] - B0x),
            "By": lploss(pred[:, 3] - B0y, target[:, 3] - B0y),
        }
        weight_sum = (
            self.delta_Bx_weight + self.delta_By_weight
            if self.use_weighted_mean
            else 1.0
        )
        loss_delta_B = (
            self.delta_Bx_weight * losses["Bx"]
            + self.delta_By_weight * losses["By"]
        ) / weight_sum
        if return_components:
            return loss_delta_B, {k: v.item() for k, v in losses.items()}
        return loss_delta_B

@register_loss("physics-informed-bfield")
def create_physics_informed_bfield_loss(params: Dict[str, Any]) -> MHDDirectBFieldLoss:
    """Create the direct-field MHD objective."""
    return MHDDirectBFieldLoss(**params)
