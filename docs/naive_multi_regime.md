# Naive multi-regime scOT ablation

This recipe reproduces the three-channel naive multi-regime ablation. It predicts `(u_x, u_y, A)` and appends standardized, spatially constant `log10(Re)` and `log10(Rm)` maps to the scOT
input. It does not use FiLM or deep adapters.

## Prerequisite

First train the corrected single-regime transfer-learning recipe:

```bash
python scripts/train_scot.py --config \
  configs/ablations/scot_with_tl/re1000.yaml
```

Its best checkpoint must be available at
`$OUTPUT_ROOT/checkpoints/scot_with_tl_re1000.pt`.

## Data layout

Set `DATA_ROOT` to a directory containing one directory per regime:

```text
$DATA_ROOT/
  mhd_Re80_N1000/mhd_data_3channel.npy
  mhd_Re200_N1000/mhd_data_3channel.npy
  ...
  mhd_Re4500_N1000/mhd_data_3channel.npy
```

Each array follows the three-channel format in `docs/data_format.md`.

## Train

```bash
export DATA_ROOT=/path/to/multi_re_data
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_scot.py --config \
  configs/ablations/naive_multi_regime/multi_re.yaml
```
