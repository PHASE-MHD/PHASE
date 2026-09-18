# Four-channel single-Re DT scOT

This recipe reproduces the best single-regime deterministic PHASE operator for
decaying turbulence at `Re=Rm=1000`. It is also the required warm start for
the four-channel multi-regime ablation.

## Data

Set `DATA_ROOT` so this file exists:

```text
$DATA_ROOT/mhd_Re1000_N1000/mhd_data_4channel.npy
```

The array must have shape `(1000, 101, 128, 128, 4)` with channel-last order
`(u_x, u_y, B_x, B_y)`. The seed-42 split is 800 train, 100 validation, and
100 test trajectories. `sub_t=4` retains 26 frames and `sub_x=1` retains
the full spatial grid.

## Model and objective

The model expands `camlab-ethz/Poseidon-T` to four outputs. Velocity uses
POSEIDON-native fluid normalization. Both magnetic input and output weights are
initialized from the mean of the pretrained velocity weights. The model learns
magnetic residuals relative to the initial magnetic field, predicts velocity
directly, and applies periodic spectral Helmholtz projection to `(u_x,u_y)`
and `(B_x,B_y)`.

Dataset scaling is `[1, 1, 4.27121774e-3, 4.27121774e-3]`; paired vector
components share one scale. The direct-field objective combines component-wise
relative L2 data and initial-condition losses, direct-B MHD PDE residuals, and
relative-L2 vorticity/current losses.

## Train

```bash
export DATA_ROOT=/path/to/four_channel_data
export OUTPUT_ROOT=/path/to/outputs
python scripts/train_scot.py --config configs/turbulence/single_re/scot_re1000.yaml
```

The canonical run starts from POSEIDON rather than an MHD checkpoint, uses
batch size 16, and targets 100 epochs. Copied parameters use
`lr=5e-6, weight_decay=1e-2`; expanded boundary tensors use
`lr=5e-4, weight_decay=0`, with the magnetic output slice receiving an
effective `lr=2e-3`. The lowest normalized validation objective is saved to
`$OUTPUT_ROOT/checkpoints/turbulence_single_re_scot_re1000.pt`.
