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

Artifact identities verified during the paranoid follow-up audit:

- historical SR full-field warm start: `133f67e0552b652ec343c4412e0db01d65f5a6e7f4814c70e89d9ef17444c67a`;
- reported MR PHASE epoch-95 checkpoint: `97c24e526430889c173472770d97389531885599696dad6d1c514dc6094fd1b4`.

Both checkpoints contain 349 model-state entries. The warm-start checkpoint
reports source epoch 90; the reported final checkpoint reports epoch 95.

## Corrected SR PHASE status

The earlier historical single-Re diffusion run reconstructed the full field.
The corrected SR PHASE recipe instead learns `[ux,uy,Bx,By]` residuals and
starts diffusion from random weights. Feature generation completed in legacy
job `179342055`; training job `179342063` completed epochs 0--99 with
exit status 0. The checkpoint selected by denormalized relative L2 was written
at epoch 90 with `denorm_loss_rel_l2=0.028242717292159797` and
`denorm_loss_mse=1.9426884546192013e-05`. Its SHA-256 digest is
`ff57dd73a0ba31a43f65acdda6cf0bb188128a71700cf39204b22bc7fcdf998e`.
This is the canonical corrected SR PHASE residual checkpoint; it is distinct
from the historical full-field checkpoint used to warm-start the reported MR
PHASE run.

## Validation contract

Regression tests lock residual construction, per-Re residual statistics,
physical-field reconstruction for metrics, full-field rather than
residual-only Helmholtz projection, strict model-only warm start, and the
single-/multi-Re config invariants. Synthetic two-epoch single-Re and
multi-Re training smokes also completed end to end in the project container;
the multi-Re smoke exercised balanced batches, per-Re paired normalization,
model-only warm start, validation, checkpointing, and held-out evaluation.
A separate one-epoch regression smoke confirmed that the guarded previous-DINO
full-field recipe still trains, validates, checkpoints, and tests unchanged.

## Paranoid follow-up audit

A post-assembly parity pass found and corrected two public-runtime differences:
PHASE had initially evaluated epoch 0 (and an unscheduled final epoch), and its
generic checkpoint `loss` field held normalized MSE. The public trainer now
matches the production PHASE schedule (`10,20,...,90` for single-Re and
`5,10,...,95` for multi-Re) and stores the selected denormalized relative L2
under `loss`. The normalized validation MSE remains available separately as
`model_val_loss`. Disabled derivative weights and optimizer parameter groups
are explicit in both public configs to avoid dependency on constructor defaults.

## Remaining scope boundaries

The public trainer covers clean training from epoch zero and strict model-only
warm starts. Full optimizer/scheduler resume across scheduler jobs is not yet
implemented and is rejected by the PHASE recipe guard. Legacy TensorBoard,
per-Re diagnostic, and plotting callbacks are also outside Batch 11; they do
not alter the training objective. The complete 102 GB production feature store
was inspected structurally but was not regenerated end to end during this
audit.
