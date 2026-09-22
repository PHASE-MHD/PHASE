# Kelvin-Helmholtz residual diffusion

single- and multi-regime PHASE diffusion recipes
for the periodic Kelvin-Helmholtz (KH) trajectories. Diffusion learns `DNS - scOT` for all four fields
`[ux, uy, Bx, By]`.

The projection is not applied to the residual alone. In physical units the implementation forms `scOT + residual`, projects the velocity and magnetic pairs separately, and converts the result back to residual form for the EDM objective.

The single-Re model starts from random diffusion weights. The multi-Re model loads only the single-Re diffusion model weights, resets epoch, optimizer, and scheduler, then trains across all ten regimes. Its scOT condition remains Re/Rm conditioned; the diffusion U-Net does not.

## Environment

```bash
export DATA_ROOT=/path/to/kh/four_channel_arrays
export FEATURE_ROOT=/path/to/generated/diffusion/features
export STATS_ROOT=/path/to/train_only/diffusion/statistics
export OUTPUT_ROOT=/path/to/training/outputs
```

## Single-Re Re=Rm=1000

```bash
python scripts/generate_scot_diffusion_features.py \
  --conditioner-config configs/kh/single_re/scot_re1000.yaml \
  --conditioner-checkpoint "$KH_SINGLE_RE_SCOT_CHECKPOINT" \
  --output-root "$FEATURE_ROOT/kh/single_re/phase_re1000" \
  --batch-size 4 \
  --num-workers 1 \
  --single-re 1000

python scripts/compute_statistics.py diffusion \
  --input "$FEATURE_ROOT/kh/single_re/phase_re1000/train" \
  --output-prefix "$STATS_ROOT/kh/single_re/phase_re1000" \
  --prediction-mode residual

python scripts/train_dino.py \
  --config configs/kh/single_re/phase_re1000.yaml
```

## Multi-Re t=[0,5]

```bash
python scripts/generate_scot_diffusion_features.py \
  --conditioner-config configs/kh/multi_re/scot_t0_5.yaml \
  --conditioner-checkpoint "$KH_MULTI_RE_SCOT_CHECKPOINT" \
  --output-root "$FEATURE_ROOT/kh/multi_re/phase_t0_5" \
  --batch-size 4 \
  --num-workers 1

python scripts/compute_statistics.py diffusion-per-re \
  --input "$FEATURE_ROOT/kh/multi_re/phase_t0_5/train" \
  --output-dir "$STATS_ROOT/kh/multi_re/phase_t0_5" \
  --prediction-mode residual

python scripts/train_dino.py \
  --config configs/kh/multi_re/phase.yaml
```

```bash
python scripts/train_dino.py \
  --config configs/kh/multi_re/phase.yaml \
  --resume-checkpoint "$OUTPUT_ROOT/checkpoints/kh_multi_re_phase_t0_5.pt"
```