## Neural-operator trajectories

The layout is:

    [sample, time, x, y, channel]

The single-regime loader converts this to the layout:

    [sample, channel, time, x, y]

Supported physical channel sets are:

- three-channel vector-potential data: ux, uy, A
- four-channel direct-field data: ux, uy, Bx, By

## Diffusion features

Diffusion data can be stored either as one NumPy object dictionary or as a
directory of contiguous arrays. The canonical keys/files are:

- diff_inputs: conditioner trajectories
- diff_targets: DNS trajectories
- re: Reynolds number for each source simulation
- sample_id: original source-simulation identifier

Five-dimensional diffusion features use:

    [sample, channel, time, x, y]

## Normalization

Statistics must be fitted on the training split only.

Paired normalization assigns a shared scale to ux/uy and another shared scale
to Bx/By. This is required when Helmholtz projection is performed in normalized
coordinates. Multi-Re scOT supports metadata-selected per-Re physical scales.
Multi-Re diffusion supports per-Re paired min-max statistics and interpolates
statistics linearly in log(Re) for unseen Reynolds numbers.

## Raw simulation conversion

Dedalus HDF5 conversion and statistics fitting are preprocessing operations,
not dataloader responsibilities. Public scripts for those operations are in
`scripts/prepare_data.py` and `scripts/compute_statistics.py`.
