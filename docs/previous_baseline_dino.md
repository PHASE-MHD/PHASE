# Previous DINO baseline

This recipe reproduces the `DINO` row in the PHASE ablation table. The
conditioning model predicts `[ux,uy,A]` at `Re=Rm=1000`; a conditional EDM
U-Net then learns the complete DNS field at each stored time. Unlike PHASE, this diffusion
baseline does not learn a residual.

The baseline intentionally has no Helmholtz projection, Re/Rm conditioning,
vorticity or current loss, or diffusion warm start. The full-field diffusion
model starts from random weights. Its architecture and optimization settings
are defined in `configs/previous_baseline/dino/re1000.yaml`.

## Obtain the conditioner

The reported baseline uses the external `tfno_Re1000.pt` checkpoint released
by the DINO authors, not the locally trained tFNO ablation described in
`docs/previous_baseline_tfno.md`. Download it from:

<https://drive.google.com/drive/folders/1hTdHoYCdW59gZYDBUgc06TdY7OHGdghi>

External checkpoints are not distributed with PHASE. The matching conditioner
architecture is recorded in
`configs/previous_baseline/dino/conditioner_re1000.yaml`.

## Prepare trajectory statistics

```bash
python scripts/compute_statistics.py trajectory \
  --input "$DATA_ROOT/mhd_Re1000_N1000/mhd_data_3channel.npy" \
  --output "$STATS_ROOT/train_only_stats_Re1000.npz" \
  --train-size 900 \
  --seed 42 \
  --split-mode single_re_seed42 \
  --sub-t 4
```

## Generate conditioning features

```bash
export DATA_ROOT=/path/to/data
export STATS_ROOT=/path/to/statistics
export FEATURE_ROOT=/path/to/dino_re1000_features
export OUTPUT_ROOT=/path/to/outputs

python scripts/generate_diffusion_features.py \
  --config configs/previous_baseline/dino/conditioner_re1000.yaml \
  --checkpoint /path/to/tfno_Re1000.pt \
  --output-root "$FEATURE_ROOT"
```

## Fit diffusion statistics

```bash
python scripts/compute_statistics.py diffusion \
  --input "$FEATURE_ROOT/train" \
  --output-prefix "$STATS_ROOT/dino_re1000" \
  --prediction-mode direct
```

This fits separate conditioner-input and full-DNS-target statistics using only
the training features.

## Train diffusion

```bash
python -m pip install -e ".[dino]"
python scripts/train_dino.py \
  --config configs/previous_baseline/dino/re1000.yaml
```

The canonical model uses a base width of 128, multipliers
`[1,2,3,5,8,12]`, eight attention heads, 32 diffusion sampling steps, batch
size 64, and 101 configured epochs. Validation runs every ten epochs, and the
selected checkpoint minimizes denormalized relative L2 error.
