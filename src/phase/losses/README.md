# Losses

Physics-informed objectives for deterministic MHD operators.

- `lp_loss.py`: sample-wise relative or absolute Lp error.
- `physics_informed.py`: objectives for `[ux,uy,A]` and `[ux,uy,Bx,By]` models.
- `loss_factory.py`: config-driven construction of the two supported MHD objectives.
- `__init__.py`: public loss exports.

Losses are evaluated after predictions and targets are returned to physical
units. Data, initial-condition, magnetic-field, vorticity, and current terms
use relative L2 errors. PDE and optional divergence residuals use mean-squared
errors. Multi-regime batches obtain viscosity and magnetic diffusivity from
their per-sample Re/Rm metadata.

KH recipes may compute spatial relative L2 independently at each time and then
average over samples and time. Helmholtz projection is a separate model-level
operation; the four-channel PHASE recipes do not use a soft divergence loss.
