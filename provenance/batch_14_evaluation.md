# Batch 14: unified held-out-test evaluation

Batch 14 replaces the run-specific legacy `evaluate_error` scripts with one
test-only physical evaluation path for tFNO, DINO, scOT, and PHASE.

Locked decisions:

- no public train/validation split switch;
- explicit `--re` selection for every report;
- physical-unit metrics after model-appropriate reconstruction;
- one Fourier derivative convention for vector-potential and direct-B models;
- turbulence metrics average spatial-snapshot errors uniformly over time and
  trajectories; KH relative errors are computed over each complete space-time
  trajectory and then averaged uniformly over trajectories;
- mean absolute log10-ratio low/high-k spectrum errors, with shells 1--8 as
  low k, matching the reported turbulence evaluations;
- full turbulence diagnostics but only field/derived/divergence diagnostics for
  KH; and
- JSON/CSV/text outputs carrying checkpoint, config, split, sample, and
  stochastic-sampling provenance.

The diffusion evaluator constructs only the test feature store and rejects
pre-flattened stores whose trajectory boundaries cannot be audited. Newly
generated DINO features persist original source simulation IDs; reports mark
older feature stores without IDs as using test-split positions.

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

## Paranoid follow-up audit

The post-assembly audit found and fixed three correctness risks:

1. KH had initially inherited the turbulence frame-relative aggregation.
   KH now computes relative errors over each complete trajectory before
   averaging samples, matching the canonical deterministic KH analysis.
2. Diffusion evaluation initially called the training loader factory and
   instantiated train, validation, and test stores. It now constructs only the
   test store and rejects flattened data with unknowable trajectory boundaries.
3. The previous-DINO feature generator did not persist original simulation
   IDs. It now writes deterministic split-ordered sample IDs, while reports
   identify legacy stores without IDs as using test-split positions.

The audit additionally added positive CLI argument guards. Direct checks in
the established Apptainer environment verified compilation, CLI rejection,
Fourier/metric regression assertions, exact normalization parity between the
full and test-only diffusion loaders, all four inference adapters, report
generation, and the epoch/state schema of the canonical DT and KH diffusion
checkpoints. The container does not include pytest, so the equivalent
assertions were executed directly rather than through the pytest runner.
