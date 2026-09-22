# evaluation

`scripts/evaluate_error.py` is the common evaluator for tFNO, DINO, scOT, and  PHASE checkpoints. It always uses the held-out test split.

## Usage

```bash
python scripts/evaluate_error.py \
  --config configs/turbulence/multi_re/phase.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --problem turbulence \
  --re 1000 \
  --output-dir results/dt_mr_phase_re1000
```
For multi-regime models, run one command per requested Reynolds number. `--re`
is required for every model, including single-Re baselines.
The command writes:

- `evaluate_error.json`: metrics, complete run provenance, and per-sample rows;
- `evaluate_error.csv`: one aggregate row;
- `evaluate_error_per_sample.csv`: auditable trajectory-level values; and
- `evaluate_error.txt`: a readable report compatible with the historical
  `evaluate_error` summaries.

## Common physical representation

Three-channel models predict `[u_x,u_y,A]`. The evaluator reconstructs
`B_x=dA/dy` and `B_y=-dA/dx` spectrally. Four-channel models already predict
`[u_x,u_y,B_x,B_y]`. For both representations, the same periodic Fourier
operators then compute
`omega=d(u_y)/dx-d(u_x)/dy`, `j=d(B_y)/dx-d(B_x)/dy`, and both divergences.

## Aggregation contract

For turbulence, relative L2, MSE, divergence, spectrum, PDF,
standard-deviation, and kurtosis errors are first computed on each spatial
snapshot, then averaged uniformly over time and finally over held-out
trajectories. Spectrum errors are the mean absolute log10 ratio between
predicted and DNS shell power. The low-wavenumber band contains shells 1--8;
the high-wavenumber band contains the remaining resolved shells.

Turbulence reports contain all metrics. KH reports contain only
relative L2, MSE, and velocity/magnetic divergence. For KH, each metric is computed over a complete space-timetrajectory and then averaged uniformly over test samples.

## Model-specific reconstruction

- tFNO and deterministic scOT outputs are denormalized with their dataset
  normalizer before any metric is computed.
- DINO samples the full normalized trajectory and then denormalizes it.
- PHASE samples a normalized residual, reconstructs `scOT + residual` in
  physical units, applies Helmholtz projection to that full field, and only
  then evaluates it against DNS.
- Per-Re PHASE normalization always receives the actual Re metadata. Unseen
  values use the same log-Re interpolation implemented by the normalizer.