# Preprocessing
Conversion and train-split statistics for MHD trajectories and diffusion features.

- `conversion.py`: converts Dedalus HDF5 outputs to NumPy arrays and converts vector potential to in-plane magnetic fields using periodic Fourier derivatives.
- `statistics.py`: computes streaming trajectory, diffusion, per-Re diffusion, and paired magnetic P99 statistics.