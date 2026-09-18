# KH multi-Re residual diffusion t=[0,5] provenance

## Canonical legacy chain

- Repository: `DINOs`
- Conditioner config:
  `MHD-World-new/configs/KH_multiRe/config_poseidon_mhd_re_finetune_KH_bfield_globalP99_t0_5_resume_epoch45_100ep_48h_denormrel.yaml`
- Conditioner checkpoint:
  `MHD-World-new/checkpoints/KH_multiRe/poseidon_mhd_re_finetune_KH_bfield_globalP99_t0_5_resume_epoch45_100ep_48h_denormrel.pt`
- Conditioner selected epoch: 55
- Initial diffusion config:
  `DINOs/configs/model_mag/KH_multiRe_diffusion_globalP99_t0_5_epoch55_no_recond_warmstart_Re1000_per_re_norm_workers8_48h/config_diffusion_KH_multiRe_globalP99_t0_5_epoch55_residual_no_recond_warmstart_Re1000_per_re_norm_workers8_48h.yaml`
- Final continuation config:
  `DINOs/configs/model_mag/KH_multiRe_diffusion_globalP99_t0_5_epoch55_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch75_48h/config_diffusion_KH_multiRe_globalP99_t0_5_epoch55_residual_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch75_48h.yaml`
- Final checkpoint:
  `DINOs/checkpoints/mag/KH_multiRe_diffusion_globalP99_t0_5_epoch55_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch75_48h/Diffusion_KH_multiRe_globalP99_t0_5_epoch55_residual_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch75_48h.pt`
- Selected checkpoint: epoch 95
- Training reached: epoch 99
- Selection metric: denormalized validation relative L2
- Selected relative L2: 0.008567048760131002
- Selected denormalized MSE: 2.1235328065884574e-05

The wall-clock continuation chain resumed full model, optimizer, scheduler, and
epoch state at epochs 20, 40, 60, and 75. The public YAML instead describes one
fresh epoch-0-to-100 experiment; generic resume is a runtime operation and is
not encoded as a different scientific recipe.

## Warm start

The initial multi-Re run loaded model weights only from the canonical
single-Re KH residual-diffusion checkpoint selected at epoch 95. Epoch,
optimizer, and scheduler were reset before multi-Re epoch 0. The architecture
is unchanged by the transition. The diffusion U-Net has no Re/Rm conditioning;
the multi-Re scOT condition and the per-Re normalization carry the regime
information.

## Features and normalization

The source feature job is
`DINOs/job_scripts/mag/KH_multiRe_diffusion/job_generate_features_KH_multiRe_globalP99_t0_5_epoch55_gpuvolta.sh`.
It exported the exact Batch 12 epoch-55 scOT condition over all ten
`Re=Rm` values. The train/validation/test stores contain 8000/1000/1000
trajectories, with 800/100/100 trajectories from every regime and shape
`[N,4,51,128,128]`.

Legacy roots:

- `DINOs/diffusion_data/mag/KH_multiRe_globalP99_t0_5_epoch55`
- `DINOs/stats_gadi/mag/KH_multiRe_globalP99_t0_5_epoch55/per_re_paired_minmax_residual_targets`

Condition and `DNS - scOT` residual statistics are fitted independently for
each Re using training features only. Each per-Re normalizer pairs
`[ux,uy]` and `[Bx,By]`. Balanced batch sampling includes all ten regimes.

## Locked training semantics

The model predicts residuals for all four fields. The reconstructed full
velocity and magnetic fields are Helmholtz projected; the residual alone is
never projected. The U-Net and EDM settings match the single-Re recipe, with
batch size 64, eight data-loader workers, AdamW at 5e-5, weight decay 1e-4,
cosine scheduling, 32 sampling steps, 100 epochs, and full validation every
fifth epoch. No diffusion vorticity/current loss is active.

## Artifact hashes

    initial config    f7ff810aa999b031f4db0c6219835b423f3e9f8ce5b5d819e35a1ffb79495b9e
    final config      0e856da21fa5cbbc750bf54316015e90aa21f01ac135f1a32f736e6b274e8081
    final checkpoint  204bdebdd99150fbbb6c7e9c2e62e594e5389fd3f6b10d38b0649bae08799d5c
