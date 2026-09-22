# Held-out visualization

`scripts/visualize.py` uses the same inference, denormalization, residual
reconstruction, full-field Helmholtz projection, and Fourier derivatives as
`scripts/evaluate_error.py`. It has no train/validation split option. Every
invocation requires an explicit Reynolds number and source test-sample ID.

Install the plotting dependency with:

```bash
python -m pip install -e ".[visualization]"
```

Requested physical times are mapped to the nearest stored frames. By default,
the command writes PNG and PDF files and records the config/checkpoint hashes,
sample ID, frame indices, plotting settings, and output-file hashes in
`visualization_manifest.json`.

## Decaying turbulence

```bash
python scripts/visualize.py \
  --config configs/turbulence/multi_re/phase.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --problem turbulence \
  --re 1000 \
  --sample-id 977 \
  --times 1.0 \
  --products fields spectra pdfs \
  --output-dir results/dt_re1000_sample977
```

Field panels use denormalized physical fields and common DNS/model Fourier
derivatives. Spectrum panels show shell power, and PDF panels use the same
frame-wise DNS RMS normalization as evaluation.

## Kelvin--Helmholtz instability

```bash
python scripts/visualize.py \
  --config configs/kh/multi_re/phase.yaml \
  --checkpoint /path/to/checkpoint.pt \
  --problem kh \
  --re 2050 \
  --sample-id 62 \
  --times 0.5 1.8 3.5 \
  --products fields tracer \
  --output-dir results/kh_re2050_sample62
```

The tracer is a post-processing diagnostic, not an ML output channel and not a
stored DNS tracer. The command initializes the same passive scalar for DNS and
model velocities, advances it with periodic semi-Lagrangian advection plus
spectral diffusion, and defaults the tracer diffusivity to `1/Re`. KH
visualization rejects turbulence-only spectra and PDFs.
