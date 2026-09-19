# Batch 14b: held-out-test visualization

Batch 14b consolidates the run-specific plotting scripts into one public,
test-only path for tFNO, DINO, scOT, and PHASE. It consumes physical
`EvaluationRecord` trajectories from Batch 14 rather than reimplementing
model loading or denormalization.

Locked behavior:

- exact Re and sample-ID selection;
- model/DNS/error panels for all four primary and two derived fields;
- one transpose from stored `[x,y]` arrays to image `[y,x]` orientation;
- DNS-derived shared field color limits and independent error limits;
- the same Fourier derivatives and shell spectra as quantitative evaluation;
- per-curve total-power normalization for displayed turbulence spectra;
- DNS-RMS normalization and evaluation histogram ranges for displayed PDFs;
- KH-only passive-tracer post-processing with periodic interpolation and
  spectral diffusion; and
- JSON provenance with plotting arguments and hashes for every generated
  artifact.

The tracer implementation is adapted from
`DINOs/analysis_scripts/plot_kh_diffusion_fields_tracer.py`. Field,
spectrum, and PDF conventions were checked against the final DINOs analysis
scripts, but inference and physics calculations use the audited Batch 14
modules.

Validation includes physical-time index mapping, exact sample filtering,
zero-velocity tracer invariance, nonempty synthetic field/spectrum/PDF/tracer
figures, orientation review, CLI guards, and compilation in the established
Apptainer environment.

Validation was run in the established project container. Source, scripts, and
tests compile; the CLI loads; exact sample selection, physical-time mapping,
tracer invariance, and all four synthetic plot products pass; and the rendered
field, spectrum, and tracer panels were inspected directly. The container does
not include `pytest`, so the committed pytest cases were exercised through
equivalent direct checks rather than through the pytest runner.

## Paranoid audit

A second source-to-artifact audit found and corrected the following release
hardening issues:

- the development extra now installs Matplotlib, which the visualization tests
  import;
- requested times must lie inside the physical interval rather than within
  half a stored-frame spacing outside it;
- non-unit domain lengths now propagate consistently to field extents, tracer
  interpolation, and tracer spectral diffusion;
- Re, domain, time, and tracer arguments reject NaN and infinity;
- exact diffusion sample filtering is regression-tested before EDM sampling;
- manifests now include requested times, formats, DPI, and worker count; and
- the KH documentation now uses source sample 62, verified to belong to the
  canonical Re=2050 held-out test split.

The canonical multi-Re KH test feature store was inspected without loading the
field arrays into memory. It has shape `[1000,4,51,128,128]`, contains all ten
Re values, and carries 100 unique source IDs for every Re. The canonical
multi-Re turbulence store has shape `[1000,4,26,128,128]`; source sample 977
was verified in its Re=1000 held-out split. All 15 public configs pass model
family, representation, and physical-time-range detection.
