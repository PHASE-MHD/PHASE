# Kelvin-Helmholtz residual diffusion

These recipes train single- and multi-regime PHASE diffusion models for
periodic Kelvin-Helmholtz trajectories over `t in [0,5]`. Diffusion learns
`DNS-scOT` for all four fields `[ux,uy,Bx,By]`.

Helmholtz projection is applied to the reconstructed physical field, not to
the residual alone. The implementation forms `scOT+residual`, projects the
velocity and magnetic pairs separately, and converts the projected correction
back to normalized residual form for the EDM objective.

## Paths

```bash
export DATA_ROOT=/path/to/kh_four_channel_data
export FEATURE_ROOT=/path/to/generated/diffusion/features
export STATS_ROOT=/path/to/train_only/diffusion/statistics
export OUTPUT_ROOT=/path/to/training/outputs
```

## Single-Re PHASE

The `Re=Rm=1000` model starts from random diffusion weights, uses paired
min-max statistics fitted on training features, trains for 100 epochs, and
validates every five epochs.

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

## Multi-Re PHASE

The multi-Re model conditions on the canonical `t in [0,5]` gated-adapter
scOT. It uses per-Re paired min-max statistics and no explicit Re/Rm
conditioning inside the diffusion U-Net. It loads only diffusion-model weights
from the selected single-Re residual PHASE checkpoint; epoch, optimizer, and
scheduler state start fresh.

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

The canonical run trains for 100 epochs, validates every five epochs, and
selects the checkpoint by denormalized relative L2. To continue an interrupted
run with optimizer, scheduler, and epoch state restored:

```bash
python scripts/train_dino.py \
  --config configs/kh/multi_re/phase.yaml \
  --resume-checkpoint /path/to/kh_multi_re_phase_checkpoint.pt
```
