# KH multi-Re scOT t=[0,5] provenance

## Canonical legacy chain

- Repository: MHD-World-new
- Initial config: configs/KH_multiRe/config_poseidon_mhd_re_finetune_KH_bfield_p99_bheavy_warm_deep_adapters_100ep_48h_tinyval.yaml
- Final config: configs/KH_multiRe/config_poseidon_mhd_re_finetune_KH_bfield_globalP99_t0_5_resume_epoch45_100ep_48h_denormrel.yaml
- Final checkpoint: checkpoints/KH_multiRe/poseidon_mhd_re_finetune_KH_bfield_globalP99_t0_5_resume_epoch45_100ep_48h_denormrel.pt
- Selected checkpoint: epoch 55
- Continuation reached: epoch 75
- Selection metric: full-validation denormalized relative L2
- Recorded selected metric: 0.025518675602041185

## Locked method

The chain uses all ten Re=Rm regimes, an 800/100/100 split per regime,
balanced ten-regime batches, t=[0,5], sub_t=5, nominal batch size 1,
global paired magnetic absolute-P99 scale 0.0806614549, POSEIDON transfer,
the epoch-95 single-Re KH warm start, and channel-gated deep Re/Rm adapters.

Full validation runs every fifth epoch and at the final epoch; it is the only
source of checkpoint selection. The locked global P99 value comes from the legacy sampled
estimator over raw indices 0:800 and all 251 frames, not the later seed-split
histogram estimator. The expanded magnetic input/output boundary tensors train
in the pretrained optimizer group at 1e-7. The legacy
magnetic_output_lr=2e-3 config entry is retained for compatibility but is
inactive with boundary_group=pretrained. A deterministically resampled
five-sample-per-Re diagnostic runs every epoch for monitoring. The public
config starts a fresh multi-Re optimizer at epoch zero;
legacy continuation configs are provenance artifacts, not public defaults.

The effective multi-Re objective uses global space-time relative L2 for all
primary fields, vorticity, and current, matching the frozen training source.


## Artifact hashes

    config     9b366a20d12eeffe28a4b09f64b604e31cf7fc3ad6c5bc8103b7ac095a02f143
    checkpoint ef3f02b6aa714d15a1c6577dfcea845ef8a57b9ca38eff48e0dc8de0313ee065
