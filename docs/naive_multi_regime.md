# Naive multi-regime scOT ablation

This recipe reproduces the three-channel naive multi-regime ablation used in
the PHASE model study. It predicts `(u_x, u_y, A)` and appends standardized,
spatially constant `log10(Re)` and `log10(Rm)` maps to the scOT patch-embedding
input. It does not use FiLM or deep adapters.

## Prerequisite

First train the corrected single-regime transfer-learning recipe:

```bash
python scripts/train_scot.py --config \
  configs/ablations/scot_with_tl/re1000.yaml
```

Its best checkpoint must be available at
`$OUTPUT_ROOT/checkpoints/scot_with_tl_re1000.pt`. Batch 8 copies only model
weights from that artifact. Multi-regime optimizer and scheduler state start
from scratch at epoch zero.

## Data layout

Set `DATA_ROOT` to a directory containing one directory per regime:

```text
$DATA_ROOT/
  mhd_Re80_N1000/mhd_data_3channel.npy
  mhd_Re200_N1000/mhd_data_3channel.npy
  ...
  mhd_Re4500_N1000/mhd_data_3channel.npy
```

Each array follows the three-channel format in `docs/data_format.md`. For every
regime, the seeded split contains 800 training, 100 validation, and 100 test
trajectories. `sub_t=4` and `sub_x=1`.

## Train

```bash
export DATA_ROOT=/path/to/multi_re_data
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_scot.py --config \
  configs/ablations/naive_multi_regime/multi_re.yaml
```

The nominal loader batch size is 1. The balanced sampler selects one trajectory
from each of the ten regimes per optimization step, giving an effective batch
of 10 and equal regime representation. The loss receives per-sample
`nu=1/Re` and `eta=1/Rm` values.

The two new conditioning channels use learning rate `1e-3`; warm-started
weights use an effective learning rate of `1e-7`. The public recipe targets
100 fresh multi-regime epochs and saves the checkpoint with the lowest
normalized validation objective. The historical artifact used for the reported
ablation is described separately in `provenance/ablations/naive_multi_regime.md`.
