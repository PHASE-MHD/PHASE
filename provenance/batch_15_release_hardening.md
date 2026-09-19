# Batch 15: release hardening

Batch 15 makes the audited research tree checkable as a public package without
changing model mathematics or locked training hyperparameters.

It adds a dependency-light YAML validator, optional path preflight, an explicit
inference-only role for the DINO conditioner, a Python 3.11/CUDA 12.1
environment, CI, a clean-checkout reproduction map, and a release checklist.
The validator imports no PyTorch, model, or dataset and can run before a GPU is
requested.

No legacy repository was modified and no training artifact was copied here.

## Validation

- all 15 public YAML files passed static validation;
- all 22 Batch 15 config tests passed under Gadi Python 3.11.7;
- the complete CPU suite passed with 141 passed, 3 dependency-based skips,
  and 9 GPU/checkpoint tests intentionally deselected;
- Python 3.11 compilation and Git whitespace checks passed; and
- a wheel built successfully as `phase-0.1.0-py3-none-any.whl`.

External-checkpoint compatibility tests and GPU smoke tests remain separate
release-checklist items because they require artifacts or hardware not needed
by this source-hardening batch.
