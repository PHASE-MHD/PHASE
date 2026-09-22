# scOT with POSEIDON transfer learning

This recipe reproduces the three-channel single-regime transfer-learning
ablation at `Re=Rm=1000`. It predicts `[ux,uy,A]`; evaluation derives `Bx`,
`By`, and current density spectrally from `A`.

The scOT backbone is initialized from the pretrained velocity channels of
`camlab-ethz/Poseidon-T`. The vector-potential input embedding is initialized
from the mean velocity embedding, while its output head is initialized to
zero. Velocity is predicted directly and `A` is learned residually. This is
transfer learning, not an MHD checkpoint warm start: training starts at epoch
zero with a new optimizer and scheduler.

The objective combines relative-L2 data and initial-condition losses, MHD PDE
residuals, a soft velocity-divergence penalty, and a relative magnetic-field
loss derived from `A`. The canonical run uses batch size 16 and 100 epochs;
all weights and optimizer settings are in
`configs/ablations/scot_with_tl/re1000.yaml`.

## Train

Follow `README_poseidon.md` to install the external scOT dependency and obtain
the POSEIDON weights.

```bash
export DATA_ROOT=/path/to/data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/ablations/scot_with_tl/re1000.yaml
```

The expected data file is
`$DATA_ROOT/mhd_Re1000_N1000/mhd_data_3channel.npy`.
