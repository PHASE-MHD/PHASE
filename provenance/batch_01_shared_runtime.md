# Batch 01: shared runtime foundations

## Scope

This batch admits the configuration helpers, activation registry, optimizer and
scheduler registries, and non-physics loss primitives. It does not yet include
datasets, normalization, PDE losses, models, trainers, or diffusion.

## Upstream provenance

- Repository: https://github.com/semihkacmaz/DINOs
- Local source tree: private legacy checkout (not distributed)
- Source commit: cf2b2c31220fee4f6f3d221f9d2b87572c3e58e0
- License: MIT; see THIRD_PARTY_NOTICES.md

Adapted paths:

- DINOs/src/activations/*.py -> src/phase/activations/
- DINOs/src/optimizers/*.py -> src/phase/optimizers/
- DINOs/src/losses/{loss_factory,lp_loss,standard,weighted}.py -> src/phase/losses/
- DINOs/src/utils/config.py -> src/phase/utils/config.py
- Relevant unit tests -> tests/unit/, with imports changed from src.* to
  phase.*

The exact source hashes captured during admission remain in the project work
record and can be regenerated from the pinned commit.

## Intentional changes

1. Public imports use phase.*; the legacy top-level package name src is not
   retained.
2. phase.losses temporarily registers only standard and weighted losses.
   Physics-informed losses will be admitted with their solver dependencies.
3. phase.utils exports only configuration functions until later utility
   batches arrive.
4. Scheduler registration now occurs before the disabled-scheduler lookup,
   fixing a direct-public-API KeyError.
5. standard_optimizers.py imports torch, which is required by its bundled Lion
   fallback but was missing upstream.

## Verification

- Python 3.11 compile check: passed.
- Direct factory/loss/optimizer/scheduler/YAML smoke assertions: passed in the
  existing project container with Python 3.10.13, PyTorch 2.1.0, and PyYAML 6.0.
- Full pytest execution: deferred because the existing container does not include
  pytest; pytest remains declared in the development dependencies.

No model or training algorithm behavior is changed in this batch.
