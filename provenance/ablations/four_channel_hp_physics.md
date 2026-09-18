# Four-channel Helmholtz and physics-loss ablation provenance

## Reported artifact

- Legacy repository: `MHD-World-new`
- Historical checkpoint: `poseidon_mhd_re_finetune_bfield_p99_bheavy_warm_deep_adapters_100ep_res10_gadi_continue48h.pt`
- Best checkpoint: epoch 91
- Selection metric: normalized validation objective
- Checkpoint loss: `11.978258913993836`
- Denormalized validation relative L2: `0.03067854103259742`
- Denormalized validation MSE: `2.7775735347404405e-05`
- Warm start: the epoch-98 single-Re four-channel checkpoint
- Evaluation split: held-out test, 100 Re=1000 trajectories

The multi-regime schedule was completed through continuation runs. The public
configuration expresses the same experiment as one fresh epoch-0-to-100 run;
continuation state is not part of the scientific recipe.

Held-out-test relative L2 errors at Re=1000 were:

```text
u_x      2.435623e-02
u_y      2.532529e-02
B_x      6.151765e-02
B_y      6.165696e-02
omega    1.143376e-01
J        1.431573e-01
```

## Architecture and optimization

The model has 32 channel-gated deep adapters plus output FiLM, 1,234 state
entries, and 22,291,436 state elements. The epoch-91 optimizer restores 844
warm-started/operator tensors at `lr=1e-7, wd=1e-2` and 324 conditioning
tensors at `lr=1e-3, wd=0`. Expanded boundary tensors belong to the
warm-started group. The single-Re warm start permits exactly 390 missing
adapter/FiLM keys.

The legacy YAML's scalar `nu`/`eta` fallback was `0.01`, but it was
inactive: all multi-regime batches carry per-sample `nu=1/Re` and
`eta=1/Rm`. The public YAML sets the physically correct Re=1000 fallback
`0.001` as well, eliminating that misleading dormant value without changing
the active historical computation.

## Artifact hashes

```text
config      f28f6463e0d3226dc8db1b4dd30ed1e4395c4288eb068fa964196d575c2c6b6f
checkpoint  c492c1c1b9a106df4bce556b01d94843995843cdf05c2705a76dbb7947ef1ea1
test report df46a2bc8a223ed3f1b272f7defc4201f77199af1396a9e0ecf551698a81d662
```
