# Evaluation

Unified held-out-test evaluation for tFNO, DINO, scOT, and PHASE.

- `inference.py`: loads checkpoints and reconstructs physical predictions.
- `physics.py`: computes Fourier derivatives, derived fields, divergence, and spectra.
- `metrics.py`: evaluates field, spectral, and distribution errors.
- `reporting.py`: writes text, JSON, and CSV reports with run provenance.

All diagnostics operate on denormalized physical fields. The common periodic
Fourier convention is

```text
Bx = dA/dy,  By = -dA/dx
omega = d(uy)/dx - d(ux)/dy
J = d(By)/dx - d(Bx)/dy
```

Turbulence metrics are averaged over snapshots and test trajectories. KH field
errors are computed over each complete space-time trajectory and then averaged
over test trajectories. See `docs/evaluation.md` for usage and metric details.
