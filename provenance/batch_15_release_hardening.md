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

## Paranoid audit follow-up

The post-batch adversarial pass mutated every top-level section in all public
recipes under both static and path-preflight validation. It fixed malformed
mapping and typed KH time-contract inputs that could previously raise Python
exceptions instead of returning validation errors. The expanded validator
suite now has 30 passing tests, 1,344 mutation calls complete without a crash,
and the ordinary CPU suite has 149 passing tests, 1 optional-dependency skip,
and 11 GPU/checkpoint tests intentionally deselected. The skipped scOT forward
test passes separately when the exact pinned POSEIDON source and its declared
PHASE dependencies are enabled; CI now installs that source revision.

All public CLIs import and expose help, all 15 recipes validate, YAML/TOML and
Python compilation pass, and a fresh wheel contains no data, checkpoints,
logs, caches, or bytecode. The wheel's validator was also executed from
outside the source checkout under Python 3.11. The Gadi login-node module
wrapper did not provide the `build` frontend, so the fresh artifact was built
with the established project container; CI installs and runs `build` in its
clean Python 3.11 job.
