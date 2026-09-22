# Previous DINO baseline

This path reproduces the `DINO` row in the PHASE ablation table. A
Re=Rm=1000 tFNO checkpoint first predicts `[ux, uy, A]` trajectories. A
conditional EDM U-Net then learns the complete DNS field at each time.

This baseline is reained as per the DINO model in Kacmaz et al. (2025) and intentionally has no Helmholtz projection, Re conditioning, vorticity/current loss, residual target, or diffusion warm start.

## 1. Obtain and verify the conditioner

Download `tfno_Re1000.pt` from the authors' checkpoint folder:

<https://drive.google.com/drive/folders/1hTdHoYCdW59gZYDBUgc06TdY7OHGdghi>

The checkpoint is external and is not distributed with PHASE.

## 2. Prepare trajectory normalization

```bash
python scripts/compute_statistics.py trajectory \
  --input "$DATA_ROOT/mhd_Re1000_N1000/mhd_data_3channel.npy" \
  --output "$STATS_ROOT/train_only_stats_Re1000.npz" \
  --train-size 900 --seed 42 --split-mode single_re_seed42 --sub-t 4
```

## 3. Generate conditioning features

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

## 4. Compute separate diffusion statistics

```bash
python scripts/compute_statistics.py diffusion \
  --input "$FEATURE_ROOT/train" \
  --output-prefix "$STATS_ROOT/dino_re1000" \
  --prediction-mode direct
```

This writes `dino_re1000_inputs.npz` and `dino_re1000_targets.npz` from the
training split only.

## 5. Train diffusion from scratch

```bash
python -m pip install -e '.[dino]'
python scripts/train_dino.py \
  --config configs/previous_baseline/dino/re1000.yaml
```
