# Batch 10: four-channel DT scOT and physics ablation

Batch 10 adds the four-channel deterministic PHASE chain: the best
single-Re Re=Rm=1000 scOT prerequisite and the reported gated multi-regime
scOT ablation with Helmholtz projection and direct-field physics losses.

Implemented items:

- `(u_x,u_y,B_x,B_y)` support without regressing `(u_x,u_y,A)` paths;
- POSEIDON velocity transfer and mean-velocity magnetic initialization;
- magnetic residual prediction and paired global physics normalization;
- periodic spectral Helmholtz projection on both vector pairs;
- direct-B MHD PDE residuals and relative-L2 vorticity/current losses;
- per-sample multi-regime transport coefficients;
- exact single-Re and multi-Re optimizer grouping;
- strict recipe guards and external checkpoint tests; and
- path-independent configs plus public training and provenance docs.

## Validation pass

The audit established the following:

- the epoch-98 single-Re checkpoint strict-loads with 20,778,018 state
  elements;
- model-only warm start introduces exactly 390 expected adapter/FiLM keys;
- the epoch-91 multi-Re checkpoint strict-loads with 22,291,436 state
  elements and optimizer groups `[844,324]` at learning rates
  `[1e-7,1e-3]`;
- the public epoch-91 model and canonical legacy model produce bit-identical
  output on a deterministic Re=80/Re=4500 batch;
- the public and legacy direct-B objectives produce bit-identical total loss,
  all 22 logged components, and every prediction gradient element;
- Helmholtz projection gives velocity and magnetic RMS divergence below
  `7e-14` in float64, preserves vector means, and commutes with paired
  normalization to `1.4e-15`;
- all ten real arrays resolve with shape `(1000,101,128,128,4)`;
- real loader splits are disjoint `8000/1000/1000`, every balanced training
  batch contains all ten regimes, and `sub_t=4` yields 26 frames; and
- the previous Batch 7 and Batch 9 checkpoints still strict-load after the
  shared model extension.

The established Apptainer image does not contain pytest. Equivalent direct
assertions were run in that image; the pytest regression and checkpoint tests
are committed for a complete development environment.
