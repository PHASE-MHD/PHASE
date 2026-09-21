# Naive multi-regime scOT provenance

## Reported artifact

- Legacy repository: `MHD-World-Re-naive`
- Initial job: `178899488`
- Continuation job: `179159775`
- Paper checkpoint: continuation-local epoch 53, effective epoch 113
- Selection metric: normalized validation loss, `4.93993993806839`
- Checkpoint denormalized relative L2: `0.03595153240114451`
- Checkpoint denormalized MSE: `4.320761030385256e-05`
- Warm start: corrected single-Re Batch 7 checkpoint, epoch 98

The initial job completed epochs 0--61 and entered epoch 62. The continuation
loaded the epoch-59 best checkpoint but reset its displayed epoch counter; its
local epoch 53 therefore corresponds to effective epoch 113. The current paper
table uses this continuation snapshot. This is a model/optimizer continuation,
not a fresh 100-epoch run, and the public YAML intentionally expresses the
clean epoch-0-to-99 recipe without reproducing the legacy counter reset.

The Re=1000 held-out test evaluation used 100 trajectories. Relative-L2 errors
were `0.02757389` for `u_x`, `0.02892533` for `u_y`, `0.2080780` for `B_x`,
`0.2128772` for `B_y`, `0.1255869` for vorticity, and `1.168437` for current.

## Legacy hashes

- Config: `5ba2c94b177be0d7dc7bc9949f1abb791322dcf4f9de8b91ad22948abd85ecdb`
- Naive conditioning model: `7919abafc9e132238b5445be9290b5f5cecde202600ccfd1244bbbe63aed7dae`
- Multi-Re dataset/sampler: `2e2e89f9dbfaee7b7e79500eebd8a12433c22e371c47d86d4f60125af5e12c29`
- Trainer: `d075efe1d28626c1c0576f2191afd0bc714b0c950852abb495bf9cf0967f29e9`
- Batch utilities: `d9b6c887094f19efb7ba66645d9f444475c1cbce8c5faac9fd38587a70d3ddcb`
- Effective-epoch-113 checkpoint: `3698689e7d3e958de1a01df84e1ecfd0006275ae14775158d20515b7ef99fa57`
- Corrected epoch-98 warm start: `b298e6762f7322c5806668224683e9f95a200359f9781c8a88bb8262800c8d8d`
- Re=1000 test report: `8a8d65fb791875cb3ac3e2f8faed70fa4960d158666a3e0bc48902dbd800fcac`

Absolute legacy paths are intentionally not part of the public runtime. They
remain recorded in the private handover and source inventories.
