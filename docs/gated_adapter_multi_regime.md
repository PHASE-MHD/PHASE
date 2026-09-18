# Gated-adapter multi-regime scOT ablation

This recipe reproduces the three-channel gated-adapter multi-regime ablation
used in the PHASE model study. It predicts `(u_x, u_y, A)` and conditions the
operator on standardized `log10(Re)` and `log10(Rm)` through zero-initialized
residual adapters in every scOT encoder and decoder block. A zero-initialized
output FiLM correction provides a second conditioning path.

## Prerequisite

First train the corrected single-regime transfer-learning recipe:

```bash
python scripts/train_scot.py --config \
  configs/ablations/scot_with_tl/re1000.yaml
```

Its best checkpoint must be available at
`$OUTPUT_ROOT/checkpoints/scot_with_tl_re1000.pt`. Only model weights are
loaded. The multi-regime optimizer, scheduler, and epoch count start fresh.
The 32 deep adapters and output conditioner are initialized to produce exactly
zero corrections, so the warm-started operator is unchanged before training.

## Data layout

Set `DATA_ROOT` to a directory containing one directory per regime:

```text
$DATA_ROOT/
  mhd_Re80_N1000/mhd_data_3channel.npy
  mhd_Re200_N1000/mhd_data_3channel.npy
  ...
  mhd_Re4500_N1000/mhd_data_3channel.npy
```

Each array follows the three-channel format in `docs/data_format.md`. Every
regime uses an independently seeded 800/100/100 train/validation/test split,
`sub_t=4`, and `sub_x=1`.

## Train

```bash
export DATA_ROOT=/path/to/multi_re_data
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_scot.py --config \
  configs/ablations/gated_adapter_multi_regime/multi_re.yaml
```

The nominal batch size is 1. The balanced sampler takes one trajectory from
each of the ten regimes per optimization step, producing an effective batch of
10. The MHD residual receives per-sample `nu=1/Re` and `eta=1/Rm`.

Warm-started operator parameters use learning rate `1e-7` and weight decay
`1e-2`. The new conditioning modules use learning rate `1e-3` and no weight
decay. The public recipe targets 100 fresh multi-regime epochs and keeps the
checkpoint with the lowest normalized validation objective. The historical
reported artifact is documented in
`provenance/batch_09_gated_adapter_multi_regime.md`.
