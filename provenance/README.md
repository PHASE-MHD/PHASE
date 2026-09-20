# Provenance

This directory records the legacy configs, selected checkpoint metadata, split
definitions, normalization statistics, and evaluation provenance for the
reported ablations and DT/KH model chains. It does not contain checkpoints or
large generated artifacts.

Implemented ablation provenance currently includes the previous tFNO/DINO
baselines, single-Re scOT without/with transfer learning, naive
multi-regime input conditioning, gated-adapter multi-regime conditioning, and
the four-channel Helmholtz/physics-loss ablation, and DT residual diffusion.
The corrected SR residual result and historical full-field MR warm start are
recorded as distinct artifacts. Run provenance also records
the four-channel single-Re DT warm-start prerequisite and the canonical
single- and multi-Re Kelvin-Helmholtz scOT and residual-diffusion chains at
t=[0,5]. Batch-level validation
records are stored as `batch_*.md`.
Batch 15 release hardening is recorded in
`batch_15_release_hardening.md`.
Canonical external checkpoint identities are recorded in
`checkpoint_manifest.yaml`; Batch 16 artifact, strict-load, and real-GPU
acceptance is recorded in `batch_16_artifact_gpu_acceptance.md`.
The reduced end-to-end DT training chain is recorded in
`batch_17_dt_acceptance.md`.
The canonical `t=[0,5]` KH acceptance chain is recorded in `batch_18_kh_acceptance.md`.
