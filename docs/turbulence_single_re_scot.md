# Four-channel single-Re turbulence scOT

This recipe trains the deterministic single-regime PHASE operator for decaying
turbulence at `Re=Rm=1000`. It predicts `[ux,uy,Bx,By]` directly and is the
weight-only warm start for the four-channel multi-regime scOT.

The model initializes its backbone from POSEIDON velocity weights, uses paired
physics normalization, learns a residual only for the magnetic channels, and
applies Helmholtz projection separately to the reconstructed velocity and
magnetic fields. Its objective includes component data and initial-condition
losses, direct magnetic-field PDE residuals, and relative-L2 vorticity and
current losses. The canonical weights are `vorticity=2` and `current=5`.

The run uses batch size 16, 100 epochs, and the exact architecture, optimizer,
normalization, and checkpoint settings in
`configs/turbulence/single_re/scot_re1000.yaml`.

## Data

The expected file is:

```text
$DATA_ROOT/mhd_Re1000_N1000/mhd_data_4channel.npy
```

## Train

```bash
export DATA_ROOT=/path/to/four_channel_data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/turbulence/single_re/scot_re1000.yaml
```
