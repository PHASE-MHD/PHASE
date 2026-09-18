# Gated-adapter multi-regime scOT provenance

## Reported artifact

- Legacy repository: `MHD-World-new`
- Historical job: `178899489.gadi-pbs`
- Best checkpoint: epoch 44
- Selection metric: normalized validation loss, `3.1360136553049087`
- Checkpoint denormalized relative L2: `0.026114497476257385`
- Checkpoint denormalized MSE: `2.6058404335799424e-05`
- Warm start: corrected single-Re Batch 7 checkpoint, epoch 98

The historical configuration targeted 100 fresh multi-regime epochs. Its
surviving runtime log contains complete summaries for epochs 0--50 and no
normal-completion marker, so it does not establish completion of all 100
epochs. The reported artifact is unambiguous: the mutable best checkpoint and
the separately preserved epoch-44 checkpoint are byte-identical.

The reported Re=1000 held-out test evaluation used 100 trajectories. Its
relative-L2 errors were `0.02081522` for `u_x`, `0.02170849` for `u_y`,
`0.1273943` for `B_x`, `0.1313130` for `B_y`, `0.1034736` for vorticity, and
`0.7133867` for current density.

## Architecture and optimizer state

The model installs 32 deep adapters across all scOT encoder and decoder blocks,
with 12 state entries per adapter, plus six output-FiLM state entries. The
single-Re warm start therefore permits exactly 390 missing state keys. Adapter
up-projections, adapter gate outputs, and the output FiLM head are all zero
initialized.

The epoch-44 artifact contains 1,234 model-state tensors and 22,289,366 total
state elements. Its AdamW state contains 1,168 entries: 844 tensors in the
warm-started group at learning rate `1e-7` and weight decay `1e-2`, and 324
conditioning tensors at learning rate `1e-3` with zero weight decay. The dummy
scheduler has `last_epoch=45`.

## Legacy hashes

- Config: `9548d0d236bc214935ec173c415ba1dd6d2dfdb7a9cbe3e8dd170a50353be6e3`
- Gated-adapter model: `05dd61e57a04044153e3293bafbd66fe1a08bedd8fd35dd8ee60c31ed4186f6f`
- Epoch-44 checkpoint: `cf814721e618601bf73abe9b4aea1bde92c689344c57ce39f8ef7dd4b0ace01f`
- Corrected epoch-98 warm start: `b298e6762f7322c5806668224683e9f95a200359f9781c8a88bb8262800c8d8d`
- Re=1000 test evaluation: `d3e0a2ab82a8ccc84a712f4b6b123a166c6dbed6ec47aa17beeaedd9de523eca`

Absolute legacy paths are intentionally excluded from the public runtime. They
remain recorded in the private handover and source inventories.
