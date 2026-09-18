# Batch 13: Kelvin-Helmholtz residual diffusion

Batch 13 adds the canonical single- and multi-regime residual-diffusion
recipes for the periodic Kelvin-Helmholtz problem over `t=[0,5]`.

Implemented items:

- four-channel residual targets `DNS - scOT` for `[ux,uy,Bx,By]`;
- exact epoch-95 single-Re and epoch-55 multi-Re Batch 12 conditioners;
- paired single-Re and per-Re paired multi-Re condition/residual normalization;
- full-field velocity and magnetic Helmholtz projection;
- the reported EDM U-Net and 32-step sampler;
- random single-Re initialization and model-weights-only multi-Re warm start;
- balanced ten-regime multi-Re batches with no diffusion-side Re conditioning;
- five-epoch full validation and denormalized-relative-L2 checkpointing;
- a separate full-state resume CLI for scheduler wall-clock continuations;
- path-independent feature generation, statistics, and training instructions;
- strict recipe guards, unit tests, and optional external checkpoint tests.

## Canonical historical artifacts

The single-Re legacy model completed epoch 99 and selected epoch 95 with
validation relative L2 `0.01637994165532291`. The multi-Re t=[0,5] chain
completed epoch 99 and selected epoch 95 with `0.008567048760131002`.
The latter used full-state continuations at epochs 20, 40, 60, and 75 after
its initial model-only warm start.

The t=[0,4] KH diffusion chain is not a public default. It remains an ablation.
The public multi-Re recipe starts scientifically at epoch 0 and may use
`--resume-checkpoint` to continue the same optimizer state across allocations.

## Monitoring boundary

The legacy jobs generated expensive prediction, PDF, and tiny diagnostic
samples during training. These do not define the optimization objective or
checkpoint metric and are not embedded in the clean training loop. Unified
held-out-test metrics, field panels, tracer plots, and movies belong to Batch
14. Consequently, exact bitwise replay of the historical stochastic trajectory
is not promised, while the architecture, objective, data contract,
normalization, warm start, optimizer, validation metric, and checkpoint schema
are locked.

## Validation pass

The Batch 13 validation pass established that:

- both new KH recipes and both existing DT residual recipes pass their guarded
  config validation;
- the public single- and multi-Re KH architecture, active optimizer fields,
  loader settings, and normalization mode match the selected legacy YAMLs;
- the single-Re feature stores have 800/100/100 trajectories and the multi-Re
  stores have 8000/1000/1000, with exactly 800/100/100 trajectories from each
  of the ten regimes; every trajectory has shape `[4,51,128,128]`;
- all 20 expected per-Re multi-Re statistics files exist, and representative
  files contain separate train-only condition and residual moments;
- the selected epoch-95 single-Re checkpoint strict-loads all 349 state keys
  into the public U-Net and records relative L2 `0.01637994165532291`;
- the selected epoch-95 multi-Re checkpoint strict-loads the same 349-key
  schema and records relative L2 `0.008567048760131002`;
- direct assertions cover both KH recipe guards, the t=[0,5]/51-frame
  contract, five-epoch validation, disabled diffusion Re conditioning, and
  full-state resume from epoch 40 to 41 with optimizer and scheduler state;
- the public training CLI exposes `--resume-checkpoint`;
- all source, scripts, and tests compile under the supported container Python;
  and
- Git whitespace checks pass.

The established Apptainer image does not include pytest, so the committed
pytest tests were mirrored by direct assertions in that runtime. The
login-node `python` is older than the repository's supported Python and
cannot compile the pre-existing walrus operator in
`scripts/verify_artifact.py`; the supported container Python compiled the
complete tree successfully.

## Paranoid re-audit

A second independent pass verified the public recipes against the complete
legacy single-Re YAML and both the initial and final multi-Re continuation
YAMLs. It also verified every feature split against the source DNS arrays,
all 22 normalization files, split disjointness, exact per-Re sample counts,
checkpoint hashes and strict 349-key loading, residual-statistics semantics,
balanced batching, full-field projection, zero mean modes, and full-state
resume behavior. No data, checkpoint, normalization, or training-objective
defect was found. The recipe guard and regression tests were strengthened so
that drift in the U-Net/EDM parameters, optimizer, loader, physical-time
metadata, epoch count, or canonical ten-Re grid now fails immediately.
