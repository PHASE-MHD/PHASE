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
- JSON provenance with hashes for every generated artifact.

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
