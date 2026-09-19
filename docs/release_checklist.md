# Public release checklist

## Source and packaging

- No datasets, checkpoints, logs, caches, or plots are tracked.
- python -m build creates source and wheel distributions.
- A clean Python 3.11 wheel installation imports phase.
- Attribution and POSEIDON dependency records are accurate.

## Recipes and tests

- phase-validate-configs configs passes.
- Each production recipe passes --check-paths after environment setup.
- Re/Rm, nu/eta, time range, representation, split, normalization, warm start,
  and checkpoint metric match provenance.
- Residual PHASE uses four fields and full-field Helmholtz projection.
- Previous DINO remains labeled as full-field diffusion.
- CPU tests, target-GPU smokes, and available checkpoint tests pass.

## Reported artifacts

- Paper results use the held-out test split unless labeled otherwise.
- Reports record checkpoint and config hashes.
- Multi-Re evaluations state the evaluated Re.
- DNS/model derivatives use one periodic spectral convention.
- Figures record source sample IDs and arguments.
- No result is called reproduced until its checkpoint/report are verified.
