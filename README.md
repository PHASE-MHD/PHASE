# PHASE

PHASE is currently being consolidated into a reproducible public research
repository for incompressible magnetohydrodynamics models.

The repository scaffold is intentionally minimal. Model source and canonical
experiment configurations will be added incrementally from the audited legacy
implementations, with checkpoint-compatibility and regression tests at each
stage.

## Current status

- Package name: `phase`
- Version: `0.1.0`
- Supported Python version: 3.11
- Packaging: setuptools with a `src/` layout
- Tests: pytest
- License: MIT

No training datasets, model checkpoints, logs, or generated analysis outputs
are tracked in Git. Small synthetic fixtures may be stored under
`tests/fixtures/`.

See `COPY_MANIFEST.md` for the audited legacy-source inventory and
`README_src_changes.md` for the rationale behind the required consolidation.
Canonical array layouts and normalization rules are documented in
`docs/data_format.md`. Data conversion and train-only statistics are
documented in `docs/preprocessing.md`.
PHASE is MIT-licensed under the project copyright in `LICENSE`. A limited
number of adapted components retain their original attribution, recorded
separately and precisely in `THIRD_PARTY_NOTICES.md` and the provenance docs.
