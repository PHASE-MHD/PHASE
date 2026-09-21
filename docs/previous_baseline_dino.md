# Previous DINO baseline

This path reproduces the `DINO` row in the PHASE ablation table. A released
Re=Rm=1000 tFNO checkpoint first predicts `[ux, uy, A]` trajectories. A
conditional EDM U-Net then learns the complete DNS field at each saved time,
not the error relative to the tFNO prediction.

This baseline intentionally has no Helmholtz projection, Re conditioning,
vorticity/current loss, residual target, or diffusion warm start.

## 1. Obtain and verify the conditioner

Download `tfno_Re1000.pt` from the authors' checkpoint folder:

<https://drive.google.com/drive/folders/1hTdHoYCdW59gZYDBUgc06TdY7OHGdghi>

The checkpoint is external and is not distributed with PHASE. Verify the exact
artifact used for the reported result:

```bash
python scripts/verify_artifact.py /path/to/tfno_Re1000.pt \
  a96152ba4dc4b341d9a336c4c619e55835e1776bda8655824f96d25398d59e7d
```

The artifact metadata reports epoch 234, but its original training job and full
training history cannot be independently reconstructed. That limitation is documented here.

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

The output pairs are tFNO predictions and DNS targets, both in physical units,
with shapes `[N,3,26,128,128]` before the diffusion dataset flattens time.
Expected trajectory counts are 900 train, 50 validation, and 50 test.

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

The canonical run uses 101 epochs indexed 0--100, batch size 64, validation
at epochs 10, 20, ..., 100, 32 sampling steps, and checkpoint selection by
denormalized relative L2. `load_checkpoint` must remain empty.
