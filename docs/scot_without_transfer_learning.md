# scOT without POSEIDON transfer learning

This recipe reproduces ablation 3 in the PHASE study. It trains a
single-regime `Re=Rm=1000` scOT on `[ux,uy,A]` from random initialization. It
does not load POSEIDON weights and does not use POSEIDON-native velocity
normalization.

The model learns residual updates for velocity and vector potential. Its
objective combines relative-L2 data and initial-condition losses, MHD PDE
residuals, a soft velocity-divergence penalty, and a relative magnetic-field
loss computed spectrally from `A`. The canonical run uses batch size 1 and 100
epochs. Exact weights and optimizer settings are in
`configs/ablations/scot_without_tl/re1000.yaml`.

## Train

Install PHASE and the external scOT architecture as described in
`README_poseidon.md`. The POSEIDON checkpoint itself is not required for this
ablation.

```bash
python -m pip install -e ".[scot]"
export DATA_ROOT=/path/to/data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/ablations/scot_without_tl/re1000.yaml
```

The expected data file is
`$DATA_ROOT/mhd_Re1000_N1000/mhd_data_3channel.npy`. Training begins at epoch
zero, and checkpoint selection follows the validation metric in the config.
