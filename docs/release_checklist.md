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

- Every canonical external checkpoint is listed in
  `provenance/checkpoint_manifest.yaml` with its byte size and SHA-256 digest.
- `scripts/verify_artifact_manifest.py` passes before checkpoint compatibility
  tests or reported-result evaluation.
- Paper results use the held-out test split unless labeled otherwise.
- Reports record checkpoint and config hashes.
- Multi-Re evaluations state the evaluated Re.
- DNS/model derivatives use one periodic spectral convention.
- Figures record source sample IDs and arguments.
- No result is called reproduced until its checkpoint/report are verified.

## Runtime acceptance

- The full ordinary CPU suite passes.
- All canonical checkpoint families strict-load through the public runtime.
- Real-CUDA forward/backward acceptance passes for tFNO, deterministic scOT,
  previous DINO full-field diffusion, and PHASE residual diffusion.
- Residual sampling is reconstructed in physical units and Helmholtz projection
  is applied to the full velocity and magnetic fields before evaluation.
