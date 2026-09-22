# Gated-adapter multi-regime scOT ablation

This recipe reproduces the three-channel gated-adapter multi-regime ablation
used in the PHASE model study. It predicts `(u_x, u_y, A)` and conditions the
operator on standardized `log10(Re)` and `log10(Rm)` through zero-initialized
residual adapters in every scOT encoder and decoder block.

## Prerequisite

First train the corrected single-regime transfer-learning recipe:

```bash
python scripts/train_scot.py --config \
  configs/ablations/scot_with_tl/re1000.yaml
```

Its best checkpoint must be available at
`$OUTPUT_ROOT/checkpoints/scot_with_tl_re1000.pt`. Only model weights are
loaded. The multi-regime optimizer, scheduler, and epoch count start fresh.

## Data layout

Set `DATA_ROOT` to a directory containing one directory per regime:

```text
$DATA_ROOT/
  mhd_Re80_N1000/mhd_data_3channel.npy
  mhd_Re200_N1000/mhd_data_3channel.npy
  ...
  mhd_Re4500_N1000/mhd_data_3channel.npy
```

## Train

```bash
export DATA_ROOT=/path/to/multi_re_data
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_scot.py --config \
  configs/ablations/gated_adapter_multi_regime/multi_re.yaml
```