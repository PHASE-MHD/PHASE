# Batch 14: unified held-out-test evaluation

Batch 14 replaces the run-specific legacy `evaluate_error` scripts with one
test-only physical evaluation path for tFNO, DINO, scOT, and PHASE.

Locked decisions:

- no public train/validation split switch;
- explicit `--re` selection for every report;
- physical-unit metrics after model-appropriate reconstruction;
- one Fourier derivative convention for vector-potential and direct-B models;
- equal averaging over spatial snapshots, time, and trajectories;
- mean absolute log10-ratio low/high-k spectrum errors, with shells 1--8 as
  low k, matching the reported turbulence evaluations;
- full turbulence diagnostics but only field/derived/divergence diagnostics for
  KH; and
- JSON/CSV/text outputs carrying checkpoint, config, split, sample, and
  stochastic-sampling provenance.

The selected conventions are based on the final paper-analysis scripts.
Run-specific plotting scripts and globally accumulated diagnostic variants are
not retained as metric implementations.

## Validation

- all source, scripts, and tests compile in the established project container;
- the command-line interface loads and exposes only held-out-test evaluation;
- analytic periodic fields verify identical Fourier derivatives for
  vector-potential and direct-B representations, near-zero divergence, and
  zero exact-prediction errors;
- a regression assertion fixes the reported mean absolute log-spectrum-ratio
  definition;
- synthetic end-to-end adapters pass for deterministic tFNO/scOT, full-field
  DINO, and residual PHASE with full-field projection; and
- report generation rejects non-test metadata and records both configuration
  and checkpoint digests.
