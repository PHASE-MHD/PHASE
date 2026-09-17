# Batch 02: data and normalization

## Scope

This batch establishes NumPy trajectories as the model-time data boundary and
adds single-Re, multi-Re, and diffusion datasets together with identity,
standard, min-max, paired min-max, per-Re paired min-max, and paired physics
normalization.

Raw Dedalus conversion and statistics fitting are intentionally deferred to
the preprocessing batch.

## Upstream provenance

### MHD-World-new working tree

- Repository commit: 9657acc85cd765e7b9eb1dd79c8892dd07f1312d
- License: MIT; see THIRD_PARTY_NOTICES.md

| Source file | SHA-256 |
| --- | --- |
| src/data/neurops_dataset.py | 9e4d77143d4fed5a463e970484dca79f918e042ceae6f21797776a076015fc9c |
| src/data/multi_re_neurops_dataset.py | 439f0985b63f248a0220fd9287a1a6a381c17b926dcd00707bd19ebab30b2492 |
| src/normalizations/physics_norm.py | 1f0caa3ace7f3fa9e59a83a0187c05ef10b8e2a4a9b6243285f4893d024cfab3 |

These hashes are authoritative because the selected behavior was present in
the legacy working tree beyond its recorded commit.

### DINOs working tree

- Repository commit: cf2b2c31220fee4f6f3d221f9d2b87572c3e58e0
- License: MIT; see THIRD_PARTY_NOTICES.md

| Source file | SHA-256 |
| --- | --- |
| src/data/diffusion_dataset.py | 53808aeef0d5e98b4c47ecf97c7755706c2c4d98291f593f36b9758f46d25328 |
| src/data/neurops_embedset.py | ffa1e5ac001d380513049dfd6b1dece4f13c3a5cc196cbea51771bc97b3a356c |
| src/normalizations/__init__.py | 606ec229183174fe8d390ac03d7ddb450848c5e446c2d53364e6971e3a6b2fed |
| src/normalizations/identity_norm.py | 7cfd65c26655a3ccb0f53543d486e5bc2f04cd7b7bbc4cd96a94750407c23ba0 |
| src/normalizations/min_max_norm.py | f7bb49deb4e52dca5b8869b0a4e46bca7d7d9c98191a1093a1923dce477a8fa0 |
| src/normalizations/normalization_factory.py | 04b36317edd96b379e6966ef057ea551d43d1fe186986a7bfc8a48108b7e08a1 |
| src/normalizations/standard_norm.py | 3d8c5cf2a02280a77b37c5d0c537e837fc181d36db83a8408456ff3dc7ee2091 |

## Canonical merge decisions

1. scOT datasets and paired physics normalization use MHD-World-new behavior.
2. Diffusion loading uses the latest DINOs behavior, including residual targets,
   directory-backed lazy arrays, Re/sample metadata, sample fractions, and
   balanced batches.
3. Paired and per-Re paired min-max implementations use the latest DINOs
   behavior, including interpolation and extrapolation in log(Re).
4. Public imports use phase.* and one explicit phase.data export surface.
5. A root-only /data/ ignore rule replaces the broad data/ pattern so Python
   source under src/phase/data remains tracked.

## Intentional correction

The admitted DINOs eager diffusion path left NumPy arrays unconverted whenever
normalization was disabled, then called the PyTorch-only permute method. PHASE
converts eager inputs and targets to float32 tensors before reshaping. A
regression test covers direct loading with residual targets and metadata.

## Verification

- Python 3.11 compilation: passed.
- No legacy src.* imports: passed.
- Single-Re split disjointness and shapes: passed.
- Multi-Re metadata and balanced sampler coverage: passed.
- Diffusion residual construction and source metadata: passed.
- Paired physics-normalization round trip: passed.
- Unseen-Re log-space min-max interpolation round trip: passed.
