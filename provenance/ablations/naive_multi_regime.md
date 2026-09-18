# Naive multi-regime scOT provenance

## Reported artifact

- Legacy repository: `MHD-World-Re-naive`
- Historical job: `178899488.gadi-pbs`
- Best checkpoint: epoch 59
- Selection metric: normalized validation loss, `5.424514629364014`
- Warm start: corrected single-Re Batch 7 checkpoint, epoch 98

The historical configuration targeted 100 fresh multi-regime epochs. Its
surviving runtime log records complete summaries for epochs 0--61 and then a
partial epoch 62; it does not record normal completion of the 100-epoch
schedule. The reported artifact is nevertheless unambiguous: the mutable best
checkpoint and the separately preserved epoch-59 checkpoint are byte-identical
and both contain epoch 59 with the validation loss reported above.

The reported Re=1000 held-out test relative-L2 errors were `0.02982249` for
`u_x`, `0.03143512` for `u_y`, `0.2266609` for `B_x`, `0.2317148` for `B_y`,
`0.1333956` for vorticity, and `1.277696` for current density.

## Legacy hashes

- Config: `5ba2c94b177be0d7dc7bc9949f1abb791322dcf4f9de8b91ad22948abd85ecdb`
- Naive conditioning model: `7919abafc9e132238b5445be9290b5f5cecde202600ccfd1244bbbe63aed7dae`
- Multi-Re dataset/sampler: `2e2e89f9dbfaee7b7e79500eebd8a12433c22e371c47d86d4f60125af5e12c29`
- Trainer: `d075efe1d28626c1c0576f2191afd0bc714b0c950852abb495bf9cf0967f29e9`
- Batch utilities: `d9b6c887094f19efb7ba66645d9f444475c1cbce8c5faac9fd38587a70d3ddcb`
- Epoch-59 checkpoint: `887d7d0507b70581194f772de2103ac01478b4c0a12ebd29bc8ea547dd368fdc`
- Corrected epoch-98 warm start: `b298e6762f7322c5806668224683e9f95a200359f9781c8a88bb8262800c8d8d`
- Paper evaluation record: `163eb0912cfd57673e771cca75d17dd8b08466060a8cc63db7fc9cf141cd3ef3`

Absolute legacy paths are intentionally not part of the public runtime. They
remain recorded in the private handover and source inventories.
