# MHD Data Generation

This directory contains the Dedalus scripts used to generate the two-dimensional
incompressible magnetohydrodynamics (MHD) trajectories used by PHASE.

## Contents

- `dedalus_mhd_turbulence_parallel.py`: freely decaying MHD turbulence.
- `dedalus_mhd_kh_parallel.py`: double-shear-layer Kelvin-Helmholtz (KH)
  instability.
- `my_random_fields.py`: periodic Gaussian random-field generator used to
  initialize the velocity streamfunction and/or magnetic vector potential.

The generators write raw Dedalus HDF5 snapshots. Conversion to the arrays used
for model training is a separate preprocessing step; see `../docs/preprocessing.md`.

## Physical System

Both scripts solve the two-dimensional incompressible MHD equations on a
periodic Fourier domain. Dedalus evolves the velocity vector `u`, pressure `p`,
out-of-plane magnetic vector potential `A`, and passive tracer `s`. The in-plane
magnetic field is reconstructed as

```text
B = curl(A z_hat),
```

which is divergence-free by construction. The nondimensional transport
coefficients are

```text
nu  = 1 / Re
eta = 1 / ReM
D   = nu / Schmidt
```

where `nu`, `eta`, and `D` are the kinematic viscosity, magnetic diffusivity,
and passive-tracer diffusivity, respectively.

The turbulence generator samples periodic Gaussian random fields for both the
velocity streamfunction and magnetic potential. The KH generator uses a
periodic double-shear velocity profile with a transverse perturbation and a
Gaussian-random magnetic potential.

## Requirements

The scripts require Python 3.10 or a compatible Python version with:

- Dedalus 3
- NumPy
- h5py
- PyTorch
- mpi4py and an MPI implementation
- FFTW
- Matplotlib
- docopt

`my_random_fields.py` must remain in the same directory as the two entry-point
scripts. PyTorch is used only to sample the initial Gaussian random fields; the
Dedalus simulations themselves run on CPUs.

The production environment used Dedalus 3.0.5 and Python 3.10. Installation of
Dedalus and its native MPI, FFTW, and HDF5 dependencies is platform-specific;
consult the official Dedalus installation documentation for the target system.

## Running Decaying Turbulence

From this directory, a single `Re=Rm=1000` trajectory on a `128 x 128` grid can
be generated with:

```bash
PBS_NCPUS=1 python dedalus_mhd_turbulence_parallel.py \
  --Nx 128 --Ny 128 \
  --Re 1000 --ReM 1000 \
  --N 1 \
  --tend 1.0 \
  --Dt 1e-3 \
  --output_dt 1e-2 \
  --output_dir outputs/turbulence_Re1000
```

This configuration saves 101 frames over `t = [0, 1]`.

## Running Kelvin-Helmholtz Instability

The canonical PHASE KH trajectories use `t = [0, 5]`, an output interval of
`0.02`, and magnetic-potential amplitude `sigma_A = 1e-2`:

```bash
PBS_NCPUS=1 python dedalus_mhd_kh_parallel.py \
  --Nx 128 --Ny 128 \
  --Re 1000 --ReM 1000 \
  --N 1 \
  --tend 5.0 \
  --Dt 1e-3 \
  --output_dt 2e-2 \
  --sigma_A 1e-2 \
  --kh_U0 1.0 \
  --kh_delta 0.05 \
  --kh_epsilon 0.01 \
  --kh_mode 1 \
  --kh_sigma 0.2 \
  --kh_seed 0 \
  --output_dir outputs/kh_Re1000
```

This configuration saves 251 frames over `t = [0, 5]`.

## Parallel Execution

Each trajectory is an independent Dedalus solve. The scripts use a Python
`multiprocessing.Pool`, with the worker count read from `PBS_NCPUS`:

```bash
PBS_NCPUS=8 python dedalus_mhd_turbulence_parallel.py --N 50 [other options]
```

If `PBS_NCPUS` is not set, the scripts use one worker. Set the number of workers
to the CPU allocation granted by the scheduler. These scripts should not also
be launched through multiple MPI ranks: parallelism across trajectories is
already managed by the multiprocessing pool.

By default, integration uses the fixed time step supplied through `--Dt` and
the fourth-order `RK443` Dedalus timestepper. Passing `--use_cfl` enables the
Dedalus CFL controller instead. Use `--skip_exists` to skip trajectories whose
output already contains the expected number of snapshots.

## Output Layout

For `--N 3 --output_dir outputs/example`, the scripts create:

```text
outputs/example/
|-- output-0/
|   `-- output-0_s1.h5
|-- output-1/
|   `-- output-1_s1.h5
`-- output-2/
    `-- output-2_s1.h5
```

The exact HDF5 filename is assigned by Dedalus. Each snapshot file contains the
following tasks:

- `velocity`: the two in-plane velocity components.
- `magnetic field`: the two in-plane magnetic-field components.
- `vector potential`: the out-of-plane scalar magnetic potential.
- `pressure`: pressure.
- `tracer`: a passive scalar used for visualization, not as a PHASE model
  channel.

PHASE models use `(u_x, u_y, B_x, B_y)` after preprocessing.

## Reproducibility Notes

- `my_random_fields.py` initializes the PyTorch random seed to zero.
- KH perturbation phases are controlled independently by `--kh_seed`.
- `Re` and `ReM` must both be specified explicitly when generating a dataset.
- The canonical multi-regime experiments use
  `Re = Rm = [80, 200, 400, 650, 1000, 1500, 2050, 2750, 3600, 4500]`.
- Output directories are overwritten by Dedalus unless existing complete
  trajectories are skipped with `--skip_exists`.

