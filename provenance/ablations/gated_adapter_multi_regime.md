# Gated-adapter multi-regime scOT provenance

## Reported artifact

- Legacy repository: `MHD-World-new`
- Initial job: `178899489`
- Continuation job: `179160463`
- Paper checkpoint: rebased effective epoch 90
- Selection metric: normalized validation loss, `2.8531549996733667`
- Checkpoint denormalized relative L2: `0.024361248968169092`
- Checkpoint denormalized MSE: `2.3291558048470052e-05`
- Warm start: corrected single-Re Batch 7 checkpoint, epoch 98

The initial job supplied epochs 0--44 of the canonical branch. The continuation
loaded the epoch-44 checkpoint and completed effective epochs 45--94 while its
legacy display counter restarted at zero. The paper table uses the best snapshot
at continuation-local epoch 45, rebased to effective epoch 90. The public YAML
expresses the clean epoch-0-to-99 recipe and does not reproduce the legacy
counter reset.

The Re=1000 held-out test evaluation used 100 trajectories. Relative-L2 errors
were `0.01928793` for `u_x`, `0.01996486` for `u_y`, `0.1130644` for `B_x`,
`0.1168025` for `B_y`, `0.09773044` for vorticity, and `0.6308411` for current.

## Architecture and optimizer state

The model installs 32 deep adapters across all scOT encoder and decoder blocks,
with 12 state entries per adapter, plus six output-FiLM state entries. The
single-Re warm start therefore permits exactly 390 missing state keys. Adapter
up-projections, adapter gate outputs, and the output FiLM head are all zero
initialized.

The effective-epoch-90 artifact contains 1,234 model-state tensors and
22,289,366 total state elements. Its AdamW state contains 1,168 entries: 844
tensors in the warm-started group at learning rate `1e-7` and weight decay
`1e-2`, and 324 conditioning tensors at learning rate `1e-3` with zero weight
decay. The dummy scheduler has `last_epoch=91`.

## Legacy hashes

- Config: `9548d0d236bc214935ec173c415ba1dd6d2dfdb7a9cbe3e8dd170a50353be6e3`
- Gated-adapter model: `05dd61e57a04044153e3293bafbd66fe1a08bedd8fd35dd8ee60c31ed4186f6f`
- Effective-epoch-90 checkpoint: `0411fbbea2a9ce565a4240674b67fb8d9d3547f5530272685993261f1216b079`
- Corrected epoch-98 warm start: `b298e6762f7322c5806668224683e9f95a200359f9781c8a88bb8262800c8d8d`
- Re=1000 test report: `bb0afe3967f0e762bea0d1d7b3bc29a7061b6977d1fd1b9dd12e99ee1af43d1b`

Absolute legacy paths are intentionally excluded from the public runtime. They
remain recorded in the private handover and source inventories.
