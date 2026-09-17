# scOT with POSEIDON transfer learning

This recipe reproduces the three-channel, single-regime transfer-learning
ablation at `Re=Rm=1000`. It predicts `[u_x,u_y,A]`; magnetic field
components and current density are derived from `A` during evaluation.

## Required inputs

- Canonical data:
  `${DATA_ROOT}/mhd_Re1000_N1000/mhd_data_3channel.npy`
- POSEIDON source and dependencies described in `README_poseidon.md`
- Hugging Face weights for `camlab-ethz/Poseidon-T`

The data array has shape `[sample,time,x,y,3]`, channel order
`[u_x,u_y,A]`, 1,000 trajectories, and 101 source frames. The locked recipe
uses the seed-42 split `800/100/100` and `sub_t=4`, yielding 26 frames.

## Training

```bash
export DATA_ROOT=/path/to/data/root
export OUTPUT_ROOT=/path/to/output/root
python scripts/train_scot.py \
  --config configs/ablations/scot_with_tl/re1000.yaml
```

Training begins at epoch zero. It does not warm-start from an MHD checkpoint.
Instead, the four fluid channels of `camlab-ethz/Poseidon-T` initialize the
scOT backbone. A fifth channel is added for `A`: its input weights are
initialized from the mean of the two velocity-input slices and its output
weights are initialized to zero. The model learns magnetic residuals while
the velocity outputs remain full-field predictions.

The data-level physics normalization is `[1,1,0.0052]`. Velocity additionally
uses POSEIDON native means and standard deviations inside the wrapper.
The locked batch size is 16. AdamW uses separate effective updates:

- copied POSEIDON parameters: learning rate `5e-6`, weight decay `1e-2`;
- expanded magnetic boundary slices: learning rate `5e-4`, no weight decay.

The loss uses data, initial-condition, MHD PDE, divergence-constraint, and
spectrally derived magnetic-field terms. The viscosity and resistivity are
both `1e-3`.

## Checkpoint selection

The best checkpoint is selected by normalized validation loss. The reported
legacy run completed epochs 0--99 and selected epoch 98. Checkpoints and data
are external artifacts and are not committed.

## Reproducibility note

The legacy YAML names `camlab-ethz/Poseidon-T` but does not pin a Hugging
Face revision. The local cache currently resolves that model to revision
`ec976ed5d25883ec9db4e486ebbeeefa9e08303b`, but the exact revision used by
the historical training job cannot be verified from its surviving artifacts.
Seed 42 fixes dataset membership. The legacy entry point did not seed
PyTorch's shuffled training order, so an independent run is not guaranteed to
recover the reported weights bit-for-bit. Artifact hashes and strict
checkpoint/output compatibility checks are recorded in
`provenance/ablations/scot_with_tl.md`.
