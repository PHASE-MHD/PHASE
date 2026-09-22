# scOT without POSEIDON transfer learning

This is ablation 3 in the PHASE model-ablation table. It trains the scOT
architecture from random initialization on the single-regime
`Re=Rm=1000` decaying-turbulence dataset. It does **not** load Poseidon-T
weights and does not apply POSEIDON-native fluid normalization.

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

## Train from scratch

```bash
export DATA_ROOT=/path/to/data
export OUTPUT_ROOT=/path/to/output/scot_without_tl
python scripts/train_scot.py \
  --config configs/ablations/scot_without_tl/re1000.yaml
```