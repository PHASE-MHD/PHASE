# Gated-adapter multi-regime scOT ablation

This recipe reproduces the three-channel gated-adapter ablation. It predicts
`[ux,uy,A]` and conditions scOT on standardized `log10(Re)` and
`log10(Rm)` through zero-initialized residual adapters in every encoder and
decoder block, followed by output FiLM modulation. Zero initialization makes
the initial conditioned network preserve the warm-start prediction.

The model covers
`Re=Rm=[80,200,400,650,1000,1500,2050,2750,3600,4500]`. Balanced batches
contain data from all ten regimes through nominal `batch_size=1` and
`res_per_batch=10`. Each regime uses an independent deterministic
800/100/100 split.

## Warm start

First train:

```bash
python scripts/train_scot.py \
  --config configs/ablations/scot_with_tl/re1000.yaml
```

Set the warm-start path in
`configs/ablations/gated_adapter_multi_regime/multi_re.yaml` to the selected
single-Re checkpoint. Only model weights are loaded. Epoch, optimizer, and
scheduler state start fresh.

## Train

Set `DATA_ROOT` to the ten three-channel regime directories described in
`docs/naive_multi_regime.md`, then run:

```bash
export DATA_ROOT=/path/to/multi_re_data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/ablations/gated_adapter_multi_regime/multi_re.yaml
```

The canonical run targets 100 epochs. Re/Rm-dependent viscosity and
resistivity are supplied per sample to the PDE residual.
