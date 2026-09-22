# Four-channel single-Re DT scOT

This recipe reproduces the best single-regime deterministic PHASE operator for
decaying turbulence at `Re=Rm=1000`. It is also the required warm start for
the four-channel multi-regime ablation.

## Data

Set `DATA_ROOT` so this file exists:

```text
$DATA_ROOT/mhd_Re1000_N1000/mhd_data_4channel.npy
```

## Train

```bash
export DATA_ROOT=/path/to/four_channel_data
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_scot.py --config configs/turbulence/single_re/scot_re1000.yaml
```