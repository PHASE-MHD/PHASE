# scOT with POSEIDON transfer learning

This recipe reproduces the three-channel, single-regime transfer-learning
ablation at `Re=Rm=1000`. It predicts `[u_x,u_y,A]`; magnetic field
components and current density are derived from `A` during evaluation.

## Required inputs
- Data:
  `${DATA_ROOT}/mhd_Re1000_N1000/mhd_data_3channel.npy`
- POSEIDON source and dependencies described in `README_poseidon.md`
- Hugging Face weights for `camlab-ethz/Poseidon-T`

## Training

```bash
export DATA_ROOT=/path/to/data/root
export OUTPUT_ROOT=/path/to/output/root
python scripts/train_scot.py \
  --config configs/ablations/scot_with_tl/re1000.yaml
```

Training begins at epoch zero. It does not warm-start from an MHD checkpoint.
Instead, the four fluid channels of `camlab-ethz/Poseidon-T` initialize the
scOT backbone. The loss uses data, initial-condition, MHD PDE, divergence-constraint, and
spectrally derived magnetic-field terms.