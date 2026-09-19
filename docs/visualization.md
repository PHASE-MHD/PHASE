# Held-out-test visualization

`scripts/visualize.py` uses the same inference, denormalization, residual
reconstruction, full-field Helmholtz projection, and Fourier derivatives as
`scripts/evaluate_error.py`. It has no train/validation split option. Every
invocation requires an explicit Reynolds number and test-sample ID.

## Turbulence

```bash
python scripts/visualize.py \\
  --config configs/turbulence/multi_re/phase.yaml \\
  --checkpoint /path/to/checkpoint.pt \\
  --problem turbulence \\
  --re 1000 \\
  --sample-id 977 \\
  --times 1.0 \\
  --products fields spectra pdfs \\
  --output-dir results/dt_re1000_sample977
```

Field panels contain model, DNS, and DNS-model columns for
`ux,uy,Bx,By,omega,J`. Model and DNS share a symmetric color scale determined
from DNS; errors use their own symmetric viridis scale. Stored arrays are
`[x,y]` and are transposed exactly once for display, so horizontal and
vertical axes correspond to physical x and y.

Spectra use the same integer-shell Fourier sums as evaluation and normalize
each plotted spectrum by its own total shell power. PDFs use the evaluation
histogram ranges and DNS RMS scales. These visual normalizations change only
the plotted curves, never the quantitative evaluation report.

## Kelvin-Helmholtz instability

```bash
python scripts/visualize.py \\
  --config configs/kh/multi_re/residual_diffusion_t0_5.yaml \\
  --checkpoint /path/to/checkpoint.pt \\
  --problem kh \\
  --re 2050 \\
  --sample-id 977 \\
  --times 0.5 1.8 3.5 \\
  --products fields tracer \\
  --output-dir results/kh_re2050_sample977
```

The tracer is reconstructed independently from the model and DNS velocity
snapshots using periodic semi-Lagrangian advection and spectral diffusion. Its
default diffusivity is `1/Re`, matching the simulation tracer equation, and
the initial profile marks the two KH shear layers. This is a post-processed
dye diagnostic, not the original Dedalus tracer state: exact reconstruction is
impossible without velocities at every internal solver step. The saved NPZ and
manifest preserve this distinction.

## Provenance

PNG and PDF are emitted by default. `visualization_manifest.json` records the
model family, held-out split, Re, exact sample-ID source, checkpoint epoch,
config/checkpoint hashes, stochastic diffusion seed and sampling steps,
physical times and frame indices, domain size, tracer parameters, and hashes
of every generated artifact. Legacy diffusion feature stores without source
IDs remain explicitly identified as test-split positions.
