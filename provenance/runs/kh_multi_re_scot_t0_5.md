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

Full validation runs every fifth epoch and is the only source of checkpoint
selection. A deterministic five-sample-per-Re diagnostic runs every epoch for
monitoring. The public config starts a fresh multi-Re optimizer at epoch zero;
legacy continuation configs are provenance artifacts, not public defaults.

See provenance/batch_12_kh_scot.md for the time-local-loss discrepancy in
the frozen historical multi-Re source.


## Artifact hashes

    config     9b366a20d12eeffe28a4b09f64b604e31cf7fc3ad6c5bc8103b7ac095a02f143
    checkpoint ef3f02b6aa714d15a1c6577dfcea845ef8a57b9ca38eff48e0dc8de0313ee065
