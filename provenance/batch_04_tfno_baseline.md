# Batch 4: previous tFNO baseline

Batch 4 adds the first complete model-training path to the clean repository:
the no-warm-start, single-Re tFNO baseline represented by `tFNO.txt`.

Included:

- the eight-layer 3-D CP-factorized tFNO architecture;
- the vector-potential MHD data, initial-condition, PDE, and divergence losses;
- Fourier differentiation and periodic MHD residual operators;
- a path-independent canonical YAML;
- a training CLI with validation-based checkpointing and held-out test metrics;
- architecture, physics, configuration, and no-warm-start tests.

Excluded intentionally:

- the downloaded tFNO checkpoint used by DINO;
- diffusion training;
- FNO, neuralop-TFNO, and UNO variants;
- scheduler/cluster launch scripts tied to Gadi.

Exact run provenance and frozen hashes are in `provenance/ablations/tfno.md`.
