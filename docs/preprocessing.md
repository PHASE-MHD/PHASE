# Data preparation

Install PHASE:

    python -m pip install -e .

The preprocessing tools use simulation data but do not include or invoke Dedalus solver. Dedalus data must be generated seperately.

## Convert Dedalus HDF5 trajectories

For direct physical fields:

    python scripts/prepare_data.py dedalus \
      --input-root /path/to/outputs \
      --output /path/to/mhd_data_4channel.npy \
      --representation direct_b

This reads tasks/velocity and tasks/magnetic field and writes channels
ux, uy, Bx, By.

For vector-potential data:

    python scripts/prepare_data.py dedalus \
      --input-root /path/to/outputs \
      --output /path/to/mhd_data_3channel.npy \
      --representation vector_potential

This reads tasks/velocity and tasks/vector potential and writes ux, uy, A.

## Convert vector potential to direct B

    python scripts/prepare_data.py vector-potential \
      --input /path/to/mhd_data_3channel.npy \
      --output /path/to/mhd_data_4channel.npy

The periodic spectral convention is Bx=dA/dy and By=-dA/dx. Domain lengths
default to Lx=Ly=1 and can be changed explicitly.

## Compute trajectory statistics

    python scripts/compute_statistics.py trajectory \
      --input /path/to/mhd_data_4channel.npy \
      --output /path/to/train_stats.npz \
      --train-size 800 \
      --seed 42

## Compute diffusion statistics

    python scripts/compute_statistics.py diffusion \
      --input /path/to/train_features.npy \
      --output-prefix /path/to/residual_stats \
      --prediction-mode residual

## Compute multi-Re magnetic p99 scales

    python scripts/compute_statistics.py multi-re-p99 \
      --data-root /path/to/multi_re_data \
      --output /path/to/magnetic_p99.json \
      --re-values 80 200 400 650 1000 1500 2050 2750 3600 4500 \
      --train-size 800 \
      --sub-t 5 \
      --split-mode multi_re_seed_plus_index