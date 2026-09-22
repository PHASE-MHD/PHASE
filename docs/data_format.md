# Data format

## Neural-operator trajectories

Canonical NumPy trajectory files use

```text
[sample, time, x, y, channel]
```

The deterministic datasets convert this layout internally to

```text
[sample, channel, time, x, y]
```

PHASE supports two physical representations:

- vector potential: `[ux, uy, A]`;
- direct magnetic field: `[ux, uy, Bx, By]`.

The direct-field convention is `Bx = dA/dy` and `By = -dA/dx`. All canonical
datasets are periodic on the unit square. Configuration files record the
physical time and spatial ranges used to construct coordinate channels.

## Dataset splits

Single-regime data use a deterministic seed-42 permutation. Multi-regime data
use an independently seeded permutation for each ordered Reynolds-number
entry. Canonical scOT datasets contain 1,000 trajectories per regime, split
into 800 training, 100 validation, and 100 held-out test trajectories. The
previous tFNO/DINO baseline uses its configured 900/50/50 split.

Normalization statistics must be fitted only on the corresponding training
indices.

## Diffusion features

Diffusion features are normally stored as a directory of contiguous arrays:

```text
diff_inputs.npy   conditioner trajectories [sample, channel, time, x, y]
diff_targets.npy  matching DNS trajectories [sample, channel, time, x, y]
sample_id.npy     source-simulation identifier [sample]
re.npy            Reynolds number [sample], for multi-regime data
```

The loader also accepts the legacy NumPy object-dictionary representation with
the same logical keys. DINO uses `diff_targets` directly. PHASE forms the clean
diffusion target as `diff_targets - diff_inputs`.

## Normalization

The deterministic scOT models use physics normalization. A shared scale for
`(ux, uy)` and another for `(Bx, By)` preserve the two vector pairs under
Helmholtz projection in normalized coordinates. The canonical deterministic
multi-regime turbulence and KH models use one global scale per paired group,
although the normalizer also supports metadata-selected per-Re scales.

Diffusion uses separate statistics for conditioner inputs and clean targets.
Multi-regime PHASE uses paired min-max statistics fitted independently at each
training Reynolds number. For an unseen Reynolds number, normalization bounds
are interpolated linearly in `log10(Re)`.

## Conversion and statistics

Use `scripts/prepare_data.py` to convert Dedalus HDF5 output or transform
vector-potential arrays to direct magnetic fields. Use
`scripts/compute_statistics.py` to fit trajectory, diffusion, and magnetic-P99
statistics. See `docs/preprocessing.md` for commands.
