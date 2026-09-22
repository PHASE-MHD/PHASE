# Naive multi-regime scOT ablation

This recipe reproduces the three-channel naive multi-regime ablation. It
predicts `[ux,uy,A]` and appends spatially constant maps containing
standardized `log10(Re)` and `log10(Rm)` to every input. It does not use FiLM
or residual adapters.

The model covers
`Re=Rm=[80,200,400,650,1000,1500,2050,2750,3600,4500]`. Balanced batches
contain data from every regime: the nominal batch size is 1 and
`res_per_batch=10`. Each regime uses its own deterministic 800/100/100 split.

## Warm start

First train:

```bash
python scripts/train_scot.py \
  --config configs/ablations/scot_with_tl/re1000.yaml
```

Set the warm-start path in
`configs/ablations/naive_multi_regime/multi_re.yaml` to the selected
single-Re checkpoint. Only model weights are loaded; the multi-regime run
starts at epoch zero with a new optimizer and scheduler.

## Data

Set `DATA_ROOT` to a directory containing:

```text
mhd_Re80_N1000/mhd_data_3channel.npy
mhd_Re200_N1000/mhd_data_3channel.npy
...
mhd_Re4500_N1000/mhd_data_3channel.npy
```

## Train

```bash
export DATA_ROOT=/path/to/multi_re_data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/ablations/naive_multi_regime/multi_re.yaml
```

The canonical run targets 100 fresh multi-regime epochs. Re/Rm-dependent
viscosity and resistivity are supplied per sample to the PDE residual.
