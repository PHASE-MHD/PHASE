# Residual diffusion for decaying turbulence

PHASE diffusion predicts a correction to a deterministic four-channel scOT
trajectory. Feature files store physical-unit scOT predictions in
`diff_inputs.npy` and matching DNS trajectories in `diff_targets.npy`. The
dataset forms the clean diffusion target lazily as `DNS - scOT`, then applies
separate train-only normalization to the condition and residual.

Both recipes use the same EDM U-Net and predict residuals for all four fields
`[ux, uy, Bx, By]`. During training and inference, the residual is first added
to the scOT condition in physical units. Helmholtz projection is then applied
to the reconstructed velocity and magnetic pairs. Projecting the residual by
itself is intentionally unsupported.

## Generate single-Re features

Train the deterministic prerequisite with
`configs/turbulence/single_re/scot_re1000.yaml`, then run:

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

This is the corrected SR PHASE recipe: paired normalization, random diffusion
initialization, and checkpoint selection by denormalized relative L2. Its
reported result remains pending until the audited rerun completes.

## Generate multi-Re features

Train the deterministic prerequisite with
`configs/ablations/four_channel_hp_physics/multi_re.yaml`, then run:

```bash
python scripts/generate_scot_diffusion_features.py \
  --conditioner-config configs/ablations/four_channel_hp_physics/multi_re.yaml \
  --conditioner-checkpoint "$MULTI_RE_SCOT_CHECKPOINT" \
  --output-root "$FEATURE_ROOT/turbulence/multi_re/phase"

python scripts/compute_statistics.py diffusion-per-re \
  --input "$FEATURE_ROOT/turbulence/multi_re/phase/train" \
  --output-dir "$STATS_ROOT/turbulence/multi_re/phase" \
  --prediction-mode residual

python scripts/train_dino.py \
  --config configs/turbulence/multi_re/phase.yaml
```

The feature exporter preserves the independent per-Re train/validation/test
splits, source sample IDs, and Reynolds numbers. The diffusion loader uses
per-Re paired min-max statistics and balanced batches containing all ten
training regimes. The diffusion U-Net itself is not Re-conditioned.

## Reported and clean warm starts

The reported MR PHASE run loaded only model weights from the historical
single-Re Re=1000 full-field diffusion checkpoint. It reset the epoch,
optimizer, and scheduler and then trained on four-field residual targets. The
public `phase.yaml` preserves that historical provenance exactly.

For a clean future experiment, replace `warm_start_checkpoint` with the
checkpoint produced by the corrected single-Re residual recipe after that run
finishes. Record that experiment as a new run; do not relabel it as the
reported MR PHASE result.
