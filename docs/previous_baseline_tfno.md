# Previous tFNO baseline

This recipe reproduces the `tFNO` row in the PHASE ablation table. It trains a
single-regime `Re=Rm=1000` tensorized Fourier neural operator on `[ux,uy,A]`
from random initialization. It is a standalone baseline and does not use
POSEIDON transfer learning or an MHD warm start.

The canonical configuration uses eight Fourier layers, width 32, eight modes
per spatial and temporal dimension, and CP factorization with rank 0.5. It
uses every fourth stored frame, a deterministic 900/50/50 train/validation/test
split, batch size 1, and 100 epochs. Architecture, optimizer, normalization,
and loss weights are defined in
`configs/previous_baseline/tfno/re1000.yaml`.

## Prepare statistics

```bash
python scripts/compute_statistics.py trajectory \
  --input "$DATA_ROOT/mhd_Re1000_N1000/mhd_data_3channel.npy" \
  --output "$STATS_ROOT/train_only_stats_Re1000.npz" \
  --train-size 900 \
  --seed 42 \
  --split-mode single_re_seed42 \
  --sub-t 4
```

Statistics are fitted on the training split only.

## Train

```bash
python -m pip install -e ".[tfno]"
export DATA_ROOT=/path/to/data
export STATS_ROOT=/path/to/statistics
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_tfno.py \
  --config configs/previous_baseline/tfno/re1000.yaml
```

The canonical config leaves `load_checkpoint` empty, so training begins at
epoch zero. The selected checkpoint minimizes `model_val_loss`.
