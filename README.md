# PHASE

PHASE is currently being consolidated into a reproducible public research
repository for incompressible magnetohydrodynamics models.

The repository is being assembled incrementally from audited legacy implementations,
with checkpoint-compatibility and regression tests at each stage. The current
package includes canonical data preprocessing, the previous tFNO and DINO
baselines, and the three-channel scOT ablation trained without POSEIDON
transfer learning.

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

## Implemented training paths

- Previous tFNO baseline, `Re=Rm=1000`, trained from scratch:
  `docs/previous_baseline_tfno.md`
- Previous DINO baseline, full-field EDM diffusion conditioned on the released
  tFNO: `docs/previous_baseline_dino.md`
- scOT without POSEIDON transfer learning, `Re=Rm=1000`, batch size 1:
  `docs/scot_without_transfer_learning.md`
