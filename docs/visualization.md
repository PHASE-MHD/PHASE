# Held-out-test visualization

`scripts/visualize.py` uses the same inference, denormalization, residual
reconstruction, full-field Helmholtz projection, and Fourier derivatives as
`scripts/evaluate_error.py`. It has no train/validation split option. Every
invocation requires an explicit Reynolds number and test-sample ID.

Install the plotting dependency before using the command:

```bash
pip install -e '.[visualization]'
```

## Turbulence

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

## Kelvin-Helmholtz instability

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