# MHD Data Generation

This directory contains the Dedalus scripts used to generate the two-dimensional
incompressible magnetohydrodynamics (MHD) trajectories used by PHASE.

## Contents

- `dedalus_mhd_turbulence_parallel.py`: freely decaying MHD turbulence.
- `dedalus_mhd_kh_parallel.py`: double-shear-layer Kelvin-Helmholtz (KH)
  instability.
- `my_random_fields.py`: periodic Gaussian random-field generator used to
  initialize the velocity streamfunction and magnetic vector potential.

## Physical System

Both scripts solve the two-dimensional incompressible MHD equations on a
periodic Fourier domain. Dedalus evolves the velocity vector `u`, pressure `p`,
out-of-plane magnetic vector potential `A`, and passive tracer `s`. The in-plane
magnetic field is reconstructed as

B = curl(A z_hat)

which is divergence-free by construction. The transport coefficients are

nu  = 1 / Re
eta = 1 / ReM

where `nu` and `eta` are the kinematic viscosity and magnetic diffusivity,
respectively.

The turbulence generator samples periodic Gaussian random fields for both the
velocity streamfunction and magnetic potential. This is adapted from Rosofsky and Huerta (2023) and Kacmaz et al. (2025). The KH generator uses a
periodic double-shear velocity profile with a transverse perturbation and a
Gaussian-random magnetic potential. `my_random_fields.py` must remain in the same directory as the two entry-point scripts.

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

The exact HDF5 filename is assigned by Dedalus. Each snapshot file contains the
following tasks:

- `velocity`: the two in-plane velocity components.
- `magnetic field`: the two in-plane magnetic-field components.
- `vector potential`: the out-of-plane scalar magnetic potential.
- `pressure`: pressure.
- `tracer`: a passive scalar used for visualization, not as a PHASE model
  channel.

PHASE models use `(u_x, u_y, B_x, B_y)` after preprocessing.
