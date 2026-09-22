# Residual diffusion for decaying turbulence

PHASE diffusion learns the correction between a deterministic four-channel
scOT trajectory and DNS for all fields `[ux,uy,Bx,By]`. Feature files store
physical-unit scOT predictions in `diff_inputs.npy` and matching DNS
trajectories in `diff_targets.npy`; the dataset forms the clean target
`DNS-scOT`.

Conditioner inputs and residual targets use separate train-only statistics.
The reconstructed full field, not the residual alone, is projected: PHASE adds
the residual to scOT in physical units, applies Helmholtz projection separately
to velocity and magnetic pairs, and converts the projected correction back to
normalized residual form for the EDM loss.

## Paths

```bash
export DATA_ROOT=/path/to/canonical/mhd/data
export FEATURE_ROOT=/path/to/generated/diffusion/features
export STATS_ROOT=/path/to/train_only/diffusion/statistics
export OUTPUT_ROOT=/path/to/training/outputs
export CHECKPOINT_ROOT=/path/to/external/historical/checkpoints
```

`CHECKPOINT_ROOT` is needed only for the exact reported multi-Re warm start.

## Single-Re PHASE

Train `configs/turbulence/single_re/scot_re1000.yaml`, then generate features,
fit residual statistics, and train diffusion:

```bash
python scripts/generate_scot_diffusion_features.py \
  --conditioner-config configs/turbulence/single_re/scot_re1000.yaml \
  --conditioner-checkpoint "$SCOT_RE1000_CHECKPOINT" \
  --output-root "$FEATURE_ROOT/turbulence/single_re/phase_re1000"

python scripts/compute_statistics.py diffusion \
  --input "$FEATURE_ROOT/turbulence/single_re/phase_re1000/train" \
  --output-prefix "$STATS_ROOT/turbulence/single_re/phase_re1000" \
  --prediction-mode residual

python scripts/train_dino.py \
  --config configs/turbulence/single_re/phase_re1000.yaml
```

This corrected single-Re recipe starts diffusion from random weights, uses
paired min-max normalization, and trains for 100 epochs with validation every
ten epochs. Checkpoint selection minimizes denormalized relative L2 error.

## Multi-Re PHASE

Train `configs/turbulence/multi_re/scot.yaml`, then run:

```bash
python scripts/generate_scot_diffusion_features.py \
  --conditioner-config configs/turbulence/multi_re/scot.yaml \
  --conditioner-checkpoint "$MULTI_RE_SCOT_CHECKPOINT" \
  --output-root "$FEATURE_ROOT/turbulence/multi_re/phase"

python scripts/compute_statistics.py diffusion-per-re \
  --input "$FEATURE_ROOT/turbulence/multi_re/phase/train" \
  --output-dir "$STATS_ROOT/turbulence/multi_re/phase" \
  --prediction-mode residual

python scripts/train_dino.py \
  --config configs/turbulence/multi_re/phase.yaml
```

The multi-Re model uses per-Re paired min-max statistics, residual targets for
all four fields, no Re/Rm conditioning inside the diffusion U-Net, 100 epochs,
and validation every five epochs. Its historical reported run loaded model
weights only from a single-Re full-field diffusion checkpoint. This warm start
does not change the multi-Re residual objective; it is retained in the
canonical config for exact reproducibility.
