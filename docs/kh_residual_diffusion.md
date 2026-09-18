# Kelvin-Helmholtz residual diffusion

Batch 13 adds the canonical single- and multi-regime PHASE diffusion recipes
for the periodic Kelvin-Helmholtz (KH) trajectories. Both consume the
deterministic conditioners from Batch 12 and use the full physical interval
`t=[0,5]`.

The feature stores contain physical-unit scOT conditions and DNS trajectories
with shape `[simulation, 4, 51, 128, 128]`. The 51 frames result from
`output_dt=0.02` and `sub_t=5`, so adjacent ML frames are separated by
`0.1`. Diffusion learns `DNS - scOT` for all four fields
`[ux, uy, Bx, By]`.

## Locked model semantics

Both recipes use the same EDM U-Net:

- `base_dim=128`, `dim_mults=[1,2,3,5,8,12]`;
- self-conditioning and eight attention heads of width 64;
- 32 denoising steps and the reported EDM noise schedule;
- no diffusion-side Re/Rm conditioning;
- no vorticity/current diffusion-loss ablation; and
- Helmholtz projection on the reconstructed full velocity and magnetic fields.

The projection is not applied to the residual alone. In physical units the
implementation forms `scOT + residual`, projects the velocity and magnetic
pairs separately, and converts the result back to residual form for the EDM
objective.

The single-Re model starts from random diffusion weights. The multi-Re model
loads only the single-Re diffusion model weights, resets epoch, optimizer, and
scheduler, then trains across all ten regimes. Its scOT condition remains
Re/Rm conditioned; the diffusion U-Net does not.

## Environment

```bash
export DATA_ROOT=/path/to/kh/four_channel_arrays
export FEATURE_ROOT=/path/to/generated/diffusion/features
export STATS_ROOT=/path/to/train_only/diffusion/statistics
export OUTPUT_ROOT=/path/to/training/outputs
```

The Batch 12 KH arrays are expected under `DATA_ROOT` with the paths encoded
by the conditioner configs. Checkpoints and generated arrays remain external
to Git.

## Single-Re Re=Rm=1000

Generate conditions with the selected epoch-95 Batch 12 checkpoint:

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
  --config configs/kh/single_re/residual_diffusion_re1000.yaml
```

The feature splits contain 800/100/100 simulations. Statistics are fitted only
to the 800 training conditions and their residual targets. Paired min-max
normalization shares one transform across `[ux,uy]` and one across
`[Bx,By]`.

## Multi-Re t=[0,5]

Generate conditions with the selected epoch-55 global-P99 Batch 12 checkpoint:

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
  --config configs/kh/multi_re/residual_diffusion_t0_5.yaml
```

Each regime contributes 800/100/100 simulations. Train-only condition and
residual statistics are computed separately for
`Re=Rm in [80,200,400,650,1000,1500,2050,2750,3600,4500]`. Batches contain
all ten regimes. Before multi-Re training, the single-Re command must have
written `$OUTPUT_ROOT/checkpoints/kh_single_re_phase_re1000.pt`; only its
model weights are loaded.

Because the multi-Re run can exceed one scheduler allocation, resume the exact
training state with:

```bash
python scripts/train_dino.py \
  --config configs/kh/multi_re/residual_diffusion_t0_5.yaml \
  --resume-checkpoint "$OUTPUT_ROOT/checkpoints/kh_multi_re_phase_t0_5.pt"
```

This restores model, optimizer, scheduler, epoch, and the incumbent best metric.
It is different from the initial model-weights-only single-Re warm start.

Both recipes perform full validation every fifth epoch and select checkpoints
by denormalized relative L2. They train for 100 epochs and use 32 diffusion
steps for validation and final held-out testing. Evaluation and KH field,
tracer, and time-evolution visualizations are consolidated separately in
Batch 14.
