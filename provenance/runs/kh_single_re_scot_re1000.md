# KH single-Re scOT provenance

## Canonical legacy run

- Repository: MHD-World
- Final config: configs/config_poseidon_mhd_finetune_KH_Re1000_bfield_p99_bheavy_alltimerel_batch1_subt5_100ep_resume_epoch58_48h.yaml
- Checkpoint: checkpoints/poseidon_mhd_finetune_KH_Re1000_bfield_p99_bheavy_alltimerel_batch1_subt5_100ep_resume_epoch58_48h.pt
- Selected checkpoint: epoch 95
- Regime: Re=Rm=1000
- Time interval: t=[0,5]
- Representation: (u_x,u_y,B_x,B_y)

## Locked method

The run uses the seed-42 800/100/100 split, sub_t=5, batch size 1,
POSEIDON velocity transfer, mean-velocity magnetic initialization, magnetic
residual prediction, paired magnetic scale 0.0669424514, Helmholtz
projection, and direct-B PDE, vorticity, and current losses. Its legacy source
applied time-local relative losses to every primary field and current density,
but not vorticity; the public recipe intentionally applies the selected
time-local method to both derived fields.

This checkpoint is the model-only warm start for multi-Re KH scOT.


## Artifact hashes

    config     a586449ab39bba2296725b9eba4de37590226d1198b9f6d6f7f13634291426fc
    checkpoint 11353c2a6400a536774b1dc9403d15fdddf7573942af3902de4edc9f7f463c47
