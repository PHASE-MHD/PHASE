# KH single-Re residual diffusion provenance

## Canonical legacy run

- Repository: `DINOs`
- Conditioner config:
  `MHD-World/configs/config_poseidon_mhd_finetune_KH_Re1000_bfield_p99_bheavy_alltimerel_batch1_subt5_100ep_resume_epoch58_48h.yaml`
- Conditioner checkpoint:
  `MHD-World/checkpoints/poseidon_mhd_finetune_KH_Re1000_bfield_p99_bheavy_alltimerel_batch1_subt5_100ep_resume_epoch58_48h.pt`
- Conditioner selected epoch: 95
- Diffusion config:
  `DINOs/configs/model_mag/singleRe_KH_diffusion_best/config_diffusion_KH_Re1000_bestscot_residual_fullfield_100ep.yaml`
- Diffusion checkpoint:
  `DINOs/checkpoints/mag/singleRe_KH_diffusion_best/Diffusion_KH_Re1000_bestscot_helmholtz_bfield_residual_fullfield_100ep.pt`
- Diffusion selected epoch: 95
- Training reached: epoch 99
- Selection metric: denormalized validation relative L2
- Selected relative L2: 0.01637994165532291
- Selected denormalized MSE: 6.0637518372459455e-05

## Features and normalization

The source feature job is
`DINOs/job_scripts/mag/singleRe_KH_diffusion_best/job_generate_diffusion_features_KH_Re1000_bestscot.sh`.
It exported physical-unit scOT predictions and DNS targets from the exact
800/100/100 seed-42 splits. Every store has shape
`[N,4,51,128,128]`, corresponding to `t=[0,5]`, `output_dt=0.02`, and
`sub_t=5`.

Legacy feature and statistics roots:

- `DINOs/diffusion_data/mag/singleRe_KH_diffusion_best/Re1000`
- `DINOs/stats/mag/singleRe_KH_diffusion_best`

The input statistics are fitted to the 800 training scOT trajectories. Target
statistics are fitted to `DNS - scOT` from those same trajectories. Paired
min-max normalization shares transforms across `[ux,uy]` and `[Bx,By]`.

## Locked training semantics

The diffusion U-Net starts from random weights and predicts residuals for all
four fields. The full field `scOT + residual` is Helmholtz projected before
the projected correction is used by the EDM objective. There is no diffusion
Re conditioning and no diffusion vorticity/current loss. Training uses batch
size 64, AdamW with learning rate 5e-5 and weight decay 1e-4, cosine scheduling,
32 diffusion sampling steps, 100 epochs, and full validation every fifth
epoch.

## Artifact hashes

    legacy config     20543bd302d98f8dddabdec38e4e2c7cde20df63e625b17c8a3b0022b1b42e63
    legacy checkpoint f919181c8a5491600abdb6c087e7c466cdc1c272bc0485ed3beb4e8e42581da3
