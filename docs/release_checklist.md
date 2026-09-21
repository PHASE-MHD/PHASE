# Public release checklist

## Source and packaging

- No datasets, checkpoints, logs, caches, or plots are tracked.
- python -m build creates source and wheel distributions.
- A clean Python 3.11 wheel installation imports phase.
- Attribution and POSEIDON dependency records are accurate.

## Recipes and tests

- Re/Rm, nu/eta, time range, representation, split, normalization, warm start,
  and checkpoint metric match the canonical config and model documentation.
- Residual PHASE uses four fields and full-field Helmholtz projection.
- Previous DINO remains labeled as full-field diffusion.
- CPU tests, target-GPU smokes, and available checkpoint tests pass.

## Reported artifacts

- Every released checkpoint has a Hugging Face model card identifying its matching
  config, selected epoch, byte size, and SHA-256 digest.
- Downloaded checkpoints are verified against their published SHA-256 before
  compatibility tests or reported-result evaluation.
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

## End-to-end acceptance

- [x] Run the reduced decaying-turbulence chain from trajectories through
  single-Re scOT, residual-feature generation, single-Re PHASE, multi-Re scOT,
  per-Re statistics, and multi-Re PHASE.
- [x] Confirm all four DT trainers reached epoch 9, wrote reloadable
  checkpoints, and produced held-out smoke metrics.
- [x] Exercise both previous-DINO full-field reconstruction and PHASE residual
  reconstruction in unit, checkpoint-compatibility, and GPU acceptance tests.
- [x] Exercise single-Re and multi-Re dispatch, paired/per-Re normalization,
  and full-field Helmholtz projection.
- [ ] Optionally rerun the reduced naive-MR and gated-adapter training smokes.
  Their current acceptance-only guards are covered by unit tests, but the first
  jobs used the pre-fix guard and the retries were terminated before training.
- [ ] Optionally complete the reduced KH chain. KH acceptance is a sanity check,
  not a release blocker; canonical KH checkpoint compatibility already passed.

These reduced experiments validate pipeline integration. They do not replace
or claim a from-scratch reproduction of every 100-epoch production result.
