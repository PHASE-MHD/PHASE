# Held-out-test evaluation

`scripts/evaluate_error.py` is the common evaluator for tFNO, DINO,
deterministic scOT, and residual-diffusion PHASE checkpoints. It deliberately
has no validation- or training-split option: public reports always use the
held-out test split constructed by the model's dataset configuration.

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
is required for every model, including single-Re baselines, so the physical
regime is never inferred from a directory name. `--max-samples` is intended
only for smoke tests; omit it for reported results. DINO and PHASE use 32 EDM
sampling steps by default and seed each trajectory with
`diffusion_seed + sample_id`, making a trajectory's stochastic prediction
independent of evaluation order.

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
No finite-difference or model-specific derivative path is used.

## Aggregation contract

Relative L2, MSE, divergence, spectrum, PDF, standard-deviation, and kurtosis
errors are first computed on each spatial snapshot. They are then averaged
uniformly over time and finally over held-out trajectories. This avoids
weighting a regime or trajectory by its field amplitude. Spectrum errors are
the mean absolute log10 ratio between predicted and DNS shell power. The
low-wavenumber band contains shells 1--8; the high-wavenumber band contains the
remaining resolved shells. This exactly preserves the reported turbulence
evaluation convention.

Turbulence reports contain all metrics. KH reports intentionally contain only
relative L2, MSE, and velocity/magnetic divergence because homogeneous
turbulence PDF and low/high-k summary metrics are not used for the instability
experiments.

## Model-specific reconstruction

- tFNO and deterministic scOT outputs are denormalized with their dataset
  normalizer before any metric is computed.
- DINO samples the full normalized trajectory and then denormalizes it.
- PHASE samples a normalized residual, reconstructs `scOT + residual` in
  physical units, applies Helmholtz projection to that full field, and only
  then evaluates it against DNS.
- Per-Re PHASE normalization always receives the actual Re metadata. Unseen
  values use the same log-Re interpolation implemented by the normalizer.

Every report stores absolute config/checkpoint paths, their SHA-256 digests,
the checkpoint epoch, exact sample IDs, Re, split, diffusion seed, and sampling
step count.
