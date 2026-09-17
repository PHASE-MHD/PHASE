# scOT without POSEIDON transfer learning

This is ablation 3 in the PHASE model-ablation table. It trains the scOT
architecture from random initialization on the single-regime
`Re=Rm=1000` decaying-turbulence dataset. It does **not** load Poseidon-T
weights and does not apply POSEIDON-native fluid normalization.

## Locked experiment

- Predicted channels: `[u_x, u_y, A]`
- Samples: 800 train, 100 validation, 100 held-out test
- Temporal subsampling: `sub_t=4`
- Dataset normalization: physics scales `[1.0, 1.0, 0.0052]`
- Prediction form: residual in velocity and vector potential
- Batch size: **1** for train, validation, and test
- Optimizer: AdamW, learning rate `5e-4`, weight decay `0`
- Scheduler: disabled
- Epochs: 100
- Checkpoint selection: minimum normalized validation loss
- Transport coefficients: `nu=eta=1e-3`

The `magnetic_input_init` and `magnetic_output_init` fields remain in the YAML
for exact legacy-config parity, but they apply only when expanding pretrained
POSEIDON weights. With transfer learning disabled, all scOT parameters,
including the magnetic pathways, use the standard random scOT initialization.

The loss combines relative-L2 data and initial-condition terms, MHD PDE
residuals, velocity-divergence regularization, and a spectral magnetic-field
loss obtained from `B=curl(A)`. The complete weights are in
`configs/ablations/scot_without_tl/re1000.yaml`.

## Install

Install PHASE and the external scOT architecture as described in
`README_poseidon.md`. Pretrained Poseidon-T weights are not downloaded for
this ablation.

```bash
python -m pip install -e ".[scot]"
```

## Data

Set `DATA_ROOT` so this file exists:

```text
${DATA_ROOT}/mhd_Re1000_N1000/mhd_data_3channel.npy
```

The array shape is `[sample, time, x, y, channel]`, with channels
`[u_x,u_y,A]`. The fixed seed-42 permutation creates the 800/100/100 split;
the final 100 samples are the test split.

## Train from scratch

```bash
export DATA_ROOT=/path/to/data
export OUTPUT_ROOT=/path/to/output/scot_without_tl
python scripts/train_scot.py \
  --config configs/ablations/scot_without_tl/re1000.yaml
```

The best checkpoint is written to
`${OUTPUT_ROOT}/checkpoints/scot_without_tl_re1000.pt`. The config guard rejects
pretrained loading, a warm start, POSEIDON-native normalization, or a training
batch size other than 1.

## Reproduce the checkpoint check

```bash
export PHASE_SCOT_WITHOUT_TL_CHECKPOINT=/path/to/checkpoint.pt
pytest -m checkpoint \
  tests/checkpoint_compatibility/test_scot_without_tl_legacy_checkpoint.py
```

The reported checkpoint was selected at epoch 95 and loads strictly into this
public implementation.
