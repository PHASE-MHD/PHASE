# Batch 5: previous DINO baseline

Batch 5 adds the second previous-study baseline used in the PHASE ablation:
released tFNO conditioning followed by full-field EDM diffusion.

Included:

- checkpoint-compatible conditional U-Net and EDM sampling/objective;
- explicit external-artifact verification;
- official tFNO state-dictionary mapping and feature generation;
- separate input/target train-only normalization;
- portable full-field DINO config and training CLI;
- tests that distinguish direct DINO targets from PHASE residual targets;
- opt-in conditioner and DINO checkpoint compatibility tests.

Excluded intentionally:

- the external 4.8 GiB model artifacts;
- PHASE residual diffusion;
- Helmholtz projection and derivative losses;
- Re conditioning;
- Gadi launch scripts and generated features.

See `docs/previous_baseline_dino.md` and `provenance/ablations/dino.md`.
