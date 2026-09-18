# DT single-Re four-channel scOT provenance

## Canonical legacy run

- Repository: `MHD-World`
- Config: `configs/config_poseidon_mhd_finetune_bfield_p99_bheavy_batch16_100ep_4h30_plots.yaml`
- Job: `173827086.gadi-pbs`
- Checkpoint: `checkpoints/poseidon_mhd_finetune_Re1000_bfield_p99_bheavy_batch16_100ep_4h30_plots.pt`
- Completed schedule: epochs 0--99
- Selected checkpoint: epoch 98
- Selection metric: normalized validation objective

Checkpoint metadata records loss `16.962203843253`, denormalized validation
relative L2 `0.037926397153309414`, and denormalized validation MSE
`3.1628246558414374e-05`. The state has 844 tensors and 20,778,018 elements.
Its optimizer groups contain 839 and five tensors at learning rates `5e-6`
and `5e-4`; the magnetic output hook gives the new output slice an effective
learning rate of `2e-3`.

## Locked method

The run uses the seed-42 800/100/100 split of the Re=Rm=1000 four-channel DT
array, `sub_t=4`, batch size 16, POSEIDON transfer learning, mean-velocity
magnetic initialization, magnetic residual prediction, paired global physics
normalization, direct-B PDE residuals, vorticity/current losses, and
full-field Helmholtz projection on velocity and magnetic pairs.

## Artifact hashes

```text
config     57bf38fcbde9a000573f0ce91ae56898e91d1535a40fbd36cea32801950d196d
checkpoint f226e9982211eb08888d8f4c6ee372b544a24fcc25ac3324dc3fe37723148ee9
```
