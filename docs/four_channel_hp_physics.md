# Four-channel Helmholtz and physics-loss ablation

This recipe reproduces the reported multi-regime scOT ablation immediately
before residual diffusion. It predicts `(u_x,u_y,B_x,B_y)`, uses the
four-channel single-Re recipe as a model-only warm start, and adds gated
Re/Rm adapters.

## Prerequisite

Train `configs/turbulence/single_re/scot_re1000.yaml` first. Its selected
checkpoint must be available at:

```text
$OUTPUT_ROOT/checkpoints/turbulence_single_re_scot_re1000.pt
```

Only model weights are loaded.

## Data

Set `DATA_ROOT` to ten directories:

```text
$DATA_ROOT/mhd_Re80_N1000/mhd_data_4channel.npy
$DATA_ROOT/mhd_Re200_N1000/mhd_data_4channel.npy
...
$DATA_ROOT/mhd_Re4500_N1000/mhd_data_4channel.npy
```

The locked `Re=Rm` values are
`[80,200,400,650,1000,1500,2050,2750,3600,4500]`. Each regime has an
independently seeded 800/100/100 split. Metadata supplies `nu=1/Re` and `eta=1/Rm` to
every sample's PDE residual.

## Train

```bash
export DATA_ROOT=/path/to/multi_re_four_channel_data
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_scot.py --config configs/turbulence/multi_re/scot.yaml
```

The run uses the same paired global normalization, direct-B objective,
magnetic residual, and final full-field Helmholtz projection as its single-Re
prerequisite. Training targets 100 fresh multi-regime epochs and
selects the lowest normalized validation objective.