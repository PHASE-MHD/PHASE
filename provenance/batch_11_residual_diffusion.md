# Batch 11: DT residual diffusion

Batch 11 exposes one residual-diffusion runtime for single- and multi-Re
PHASE while preserving the previous DINO full-field baseline as a separately
guarded recipe.

## Canonical legacy sources

- residual dataset and reconstruction:
  `DINOs/src/data/diffusion_dataset.py` and
  `DINOs/src/utils/diffusion_tensor_normalization.py`;
- EDM and full-field residual projection:
  `DINOs/src/diffusion/models/elucidated_diffusion.py`;
- multi-Re feature generation:
  `DINOs/data_generation/diffusion/generate_diffusion_features_poseidon_multi_re_bfield.py`;
- corrected SR config:
  `DINOs/configs/model_mag/singleRe_DT_SR_PHASE_residual/config_diffusion_DT_Re1000_SR_PHASE_residual_fullfield_100ep.yaml`;
- reported MR configs and checkpoint manifest:
  `DINOs/best_runs/decaying_turbulence_multi_re_scot_diffusion_100ep/`.

## Reported MR PHASE provenance

The deterministic conditioner is the best four-channel multi-Re scOT
checkpoint at epoch 91. Diffusion uses four-field residual targets, per-Re
paired min-max normalization, full-field Helmholtz projection, no diffusion
Re conditioning, batch size 64, AdamW with learning rate `5e-5` and weight
decay `1e-4`, and 32 sampling steps.

The initial diffusion weights came from the historical Re=1000 single-Re
full-field diffusion checkpoint at
`DINOs/checkpoints/mag/poseidon_bfield_p99_bheavy/Diffusion_MHDmag_Re1000_helmholtz_poseidon_bfield_p99_bheavy_100ep.pt`. Exactly model weights were loaded; optimizer,
scheduler, and epoch state were not restored. The multi-Re run began at epoch
zero and its reported best checkpoint is epoch 95 with validation
denormalized relative L2 `0.01023141382` and denormalized MSE
`1.7980645e-06`.

## Corrected SR PHASE status

The earlier historical single-Re diffusion run reconstructed the full field.
The corrected SR PHASE recipe instead learns `[ux,uy,Bx,By]` residuals and
starts diffusion from random weights. Feature generation completed in legacy
job `179342055`; the canonical training result was still pending when this
batch was assembled. No corrected SR metric or checkpoint is claimed here.

## Validation contract

Regression tests lock residual construction, per-Re residual statistics,
physical-field reconstruction for metrics, full-field rather than
residual-only Helmholtz projection, strict model-only warm start, and the
single-/multi-Re config invariants. Synthetic one-epoch single-Re and
multi-Re training smokes also completed end to end in the project container;
the multi-Re smoke exercised balanced batches, per-Re paired normalization,
model-only warm start, validation, checkpointing, and held-out evaluation.
A separate one-epoch regression smoke confirmed that the guarded previous-DINO
full-field recipe still trains, validates, checkpoints, and tests unchanged.
