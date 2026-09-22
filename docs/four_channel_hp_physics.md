# Four-channel Helmholtz and physics-loss ablation

This recipe reproduces the deterministic multi-regime PHASE ablation
immediately before diffusion. It predicts `[ux,uy,Bx,By]` directly, uses
gated Re/Rm adapters, and warm-starts model weights from the single-Re
four-channel turbulence scOT.

The model uses one global paired normalization scale across all ten regimes,
learns a residual only for the magnetic channels, and applies Helmholtz
projection separately to the complete velocity and magnetic fields. The
objective includes component data and initial-condition losses, direct
magnetic-field PDE residuals, and relative-L2 vorticity and current losses
with weights 2 and 5.

## Warm start

Train `configs/turbulence/single_re/scot_re1000.yaml` first and set the
multi-Re warm-start path to its selected checkpoint. Only model weights are
loaded; the multi-Re run begins at epoch zero with a fresh optimizer and
scheduler.

## Data

Set `DATA_ROOT` to ten four-channel datasets:

```text
mhd_Re80_N1000/mhd_data_4channel.npy
mhd_Re200_N1000/mhd_data_4channel.npy
...
mhd_Re4500_N1000/mhd_data_4channel.npy
```

The locked regimes are
`Re=Rm=[80,200,400,650,1000,1500,2050,2750,3600,4500]`. Each has an
independently seeded 800/100/100 split. Metadata supplies `nu=1/Re` and
`eta=1/Rm` to each PDE residual.

## Train

```bash
export DATA_ROOT=/path/to/multi_re_four_channel_data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/turbulence/multi_re/scot.yaml
```

Balanced batches include all ten regimes through nominal `batch_size=1` and
`res_per_batch=10`. Training targets 100 fresh epochs, and the selected
checkpoint minimizes the normalized validation objective.
