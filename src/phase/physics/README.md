# Physics Operators
differentiable Fourier-space operators for two-dimensional incompressible MHD

- `pde_solvers.py`: evaluates momentum and induction-equation residuals. It retains the `(u_x, u_y, A)` formulation used by the tFNO and early scOT ablations and provides the direct `(u_x, u_y, B_x, B_y)` formulation used by PHASE.
- `constraints.py`: evaluates velocity and magnetic divergence constraints for the vector-potential formulation.