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

The optimizer was also audited at the gradient-slice level. In the single-Re
stage, copied input/output slices have effective learning rate `5e-6`, the new
magnetic input slices have effective learning rate `5e-4`, and the new
magnetic output slices have effective learning rate `2e-3`. In the multi-Re
stage, the canonical `boundary_group: pretrained` assignment places both
copied and expanded boundary slices at `1e-7`; adapters and output FiLM use
`1e-3`. Thus, the retained multi-Re `magnetic_output_lr: 2e-3` entry is dormant
historical configuration metadata, not an active third optimizer rate.

The historical training launchers fixed data-split and balanced-sampler seeds
but did not fix PyTorch model-initialization/CUDA seeds. The reported weights
also passed through continuation jobs, whose newly constructed balanced
samplers restarted their epoch counters. The public fresh epoch-0-to-100
recipe therefore reproduces the method and checkpoint schema, but is not
expected to regenerate the historical checkpoint bit for bit.

The established Apptainer image does not contain pytest. Equivalent direct
assertions were run in that image; the pytest regression and checkpoint tests
are committed for a complete development environment.
