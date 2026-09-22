# Data preparation

Install PHASE before running the preprocessing commands:

```bash
python -m pip install -e .
```

The preprocessing package reads completed simulations; it does not invoke
Dedalus. Generate turbulence or Kelvin--Helmholtz trajectories separately with
the scripts in `MHD_DataGeneration/`.

## Convert Dedalus HDF5 trajectories

For direct physical fields:

```bash
python scripts/prepare_data.py dedalus \
  --input-root /path/to/outputs \
  --output /path/to/mhd_data_4channel.npy \
  --representation direct_b
```

The converter reads `tasks/velocity` and `tasks/magnetic field` and writes
`[ux, uy, Bx, By]` in `[sample,time,x,y,channel]` order.

For vector-potential data:

```bash
python scripts/prepare_data.py dedalus \
  --input-root /path/to/outputs \
  --output /path/to/mhd_data_3channel.npy \
  --representation vector_potential
```

This reads `tasks/velocity` and `tasks/vector potential` and writes
`[ux, uy, A]`. The input root must contain one HDF5 file in each `output-N`
directory. Existing outputs are protected unless `--overwrite` is supplied.

## Convert vector potential to direct B

```bash
python scripts/prepare_data.py vector-potential \
  --input /path/to/mhd_data_3channel.npy \
  --output /path/to/mhd_data_4channel.npy
```

The periodic Fourier convention is `Bx = dA/dy`, `By = -dA/dx`. Domain lengths
default to `Lx=Ly=1` and can be changed with `--lx` and `--ly`.

## Fit trajectory statistics

Statistics are computed from deterministic training indices only. Match
`--train-size`, `--split-mode`, `--sub-t`, and `--sub-x` to the model config.
For example, the previous tFNO/DINO conditioner uses:

```bash
python scripts/compute_statistics.py trajectory \
  --input /path/to/mhd_data_3channel.npy \
  --output /path/to/train_only_stats_Re1000.npz \
  --train-size 900 \
  --seed 42 \
  --split-mode single_re_seed42 \
  --sub-t 4
```

The output contains per-channel mean, standard deviation, minimum, maximum,
and the selected training indices.

## Fit diffusion statistics

For residual PHASE features:

```bash
python scripts/compute_statistics.py diffusion \
  --input /path/to/features/train \
  --output-prefix /path/to/statistics/phase \
  --prediction-mode residual
```

This writes separate input and residual-target statistics. Use
`--prediction-mode direct` for the full-field DINO baseline. For multi-regime
PHASE, fit each Reynolds number separately:

```bash
python scripts/compute_statistics.py diffusion-per-re \
  --input /path/to/features/train \
  --output-dir /path/to/statistics/per_re \
  --prediction-mode residual
```

The feature directory must contain `re.npy` for the per-Re command.

## Compute paired magnetic P99 scales

```bash
python scripts/compute_statistics.py multi-re-p99 \
  --data-root /path/to/multi_re_data \
  --output /path/to/magnetic_p99.json \
  --re-values 80 200 400 650 1000 1500 2050 2750 3600 4500 \
  --train-size 800 \
  --sub-t 5 \
  --split-mode multi_re_seed_plus_index
```

For each regime and for the pooled training data, the command estimates the
absolute 99th percentile of `Bx` and `By` with a streaming histogram. The
paired scale is the larger component percentile. Use `--time-stop-index` when
statistics must be restricted to a shorter time interval.
