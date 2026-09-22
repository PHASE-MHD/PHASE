# Evaluation

`scripts/evaluate_error.py` provides one held-out-test evaluator for tFNO,
DINO, deterministic scOT, and PHASE. It has no train/validation split option.

## Usage

```bash
python scripts/evaluate_error.py \
  --config configs/turbulence/multi_re/phase.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --problem turbulence \
  --re 1000 \
  --output-dir results/dt_mr_phase_re1000
```

`--re` is required for both single- and multi-regime models. Run one command
per requested Reynolds number. Diffusion defaults to 32 sampling steps and a
seed of 42; both are recorded in the report and can be overridden explicitly.

The command writes:

- `evaluate_error.json`: aggregate, per-Re, and per-sample metrics plus run
  provenance;
- `evaluate_error.csv`: one aggregate row;
- `evaluate_error_per_sample.csv`: trajectory-level metrics;
- `evaluate_error.txt`: a readable summary compatible with the historical
  reports.

## Common physical representation

Three-channel models predict `[ux,uy,A]`; evaluation reconstructs
`Bx=dA/dy` and `By=-dA/dx` spectrally. Four-channel models predict
`[ux,uy,Bx,By]` directly. The same periodic Fourier operators then compute

```text
omega = d(uy)/dx - d(ux)/dy
J     = d(By)/dx - d(Bx)/dy
```

Velocity and magnetic divergence use the same derivatives. All diagnostics
are evaluated after conversion to physical units.

## Turbulence metrics

For each held-out trajectory, the evaluator computes every metric on each
spatial frame, averages uniformly over time, and then averages uniformly over
trajectories. Reported quantities are:

- relative L2 and MSE for `ux`, `uy`, `Bx`, `By`, `omega`, and `J`;
- velocity and magnetic divergence MSE and RMS;
- low- and high-wavenumber spectrum errors for velocity, magnetic field,
  vorticity, and current;
- PDF relative MAE, relative standard-deviation error, and absolute kurtosis
  error for all primary and derived fields.

Spectrum error is the mean absolute base-10 logarithmic ratio of predicted to
DNS shell power. Low wavenumbers are shells 1--8; the high-wavenumber band is
the remaining complete shells. PDF inputs are normalized frame by frame: both
velocity components share the DNS vector RMS, both magnetic components share
the DNS magnetic-vector RMS, and vorticity/current use their own DNS RMS.

## Kelvin--Helmholtz metrics

KH reports intentionally omit turbulence spectra and distribution metrics.
Relative L2, MSE, and divergence are computed over each complete space-time
trajectory and then averaged uniformly over held-out trajectories.

## Model-specific reconstruction

- tFNO and deterministic scOT are denormalized with their dataset normalizer.
- DINO samples a full normalized trajectory and then denormalizes it.
- PHASE samples a normalized residual, reconstructs `scOT + residual` in
  physical units, and applies Helmholtz projection separately to the complete
  velocity and magnetic pairs before evaluation.
- Per-Re PHASE normalization receives the actual Re metadata. Unseen values
  use the same log-Re interpolation as the training normalizer.

Reports include configuration and checkpoint paths and SHA-256 digests, the
selected checkpoint epoch when available, the model representation, source
sample IDs, and diffusion sampling metadata.
