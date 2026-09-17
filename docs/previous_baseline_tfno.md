# Previous tFNO baseline

This package reproduces the `tFNO` row in the PHASE ablation table. It is a
single-regime `Re=Rm=1000` model over `[ux, uy, A]`, trained from random
initialization. It is not the downloaded tFNO checkpoint used to condition the
DINO baseline.

## Prepare inputs

Start from the canonical trajectory array described in `docs/data_format.md`.
For this baseline it must have shape `[1000, time, 128, 128, 3]` and channels
`[ux, uy, A]`. Compute train-only min/max statistics with the exact seed-42
split:

```bash
python scripts/compute_statistics.py \
  --input "$DATA_ROOT/mhd_Re1000_N1000/mhd_data_3channel.npy" \
  --output "$STATS_ROOT/train_only_stats_Re1000.npz" \
  --train-size 900 --seed 42 --split-mode single_re_seed42
```

## Train

```bash
python -m pip install -e '.[tfno]'
export DATA_ROOT=/path/to/data
export STATS_ROOT=/path/to/statistics
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_tfno.py \
  --config configs/previous_baseline/tfno/re1000.yaml
```

The canonical configuration intentionally keeps `load_checkpoint: ""`.
Training therefore starts at epoch zero without a warm start. The best
checkpoint is selected by normalized validation loss, matching the reported
legacy run.
