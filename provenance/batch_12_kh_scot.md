# Batch 12: Kelvin-Helmholtz deterministic scOT

Batch 12 adds the canonical single- and multi-regime deterministic scOT
recipes for the periodic Kelvin-Helmholtz problem.

Implemented items:

- four-channel KH configs for Re=Rm=1000 and all ten regimes;
- canonical t=[0,5], sub_t=5, and 51-frame physical-time handling;
- paired single-Re and global multi-Re magnetic absolute-P99 scales;
- opt-in per-time-slice relative L2 losses for all four primary fields,
  vorticity, and current;
- exact single-Re warm start and gated Re/Rm adapter behavior;
- full-validation checkpoint selection every five epochs;
- deterministic five-sample-per-Re monitoring every epoch;
- strict recipe guards, synthetic regression tests, and external checkpoint
  compatibility tests.

## Historical implementation discrepancy

The selected single-Re legacy source implements the configured time-local
losses. The frozen MHD-World-new source used by the historical multi-Re
t=[0,5] run accepted the same YAML keys through kwargs but did not apply
them. The public recipe implements the intended method selected during the KH
experiments: all four primary fields plus both derived fields use time-local
relative L2. Consequently, its model architecture and checkpoint schema are
compatible with the historical checkpoint, but a fresh training trajectory is
not expected to reproduce the historical weights bit for bit.

The canonical historical multi-Re result remains the global-P99 t=[0,5]
chain. The t=[0,4] experiments are retained only as documented ablations and
are not public defaults.

Visualization is not part of the training loop. Batch 14 is responsible for
held-out-test fields, tracer panels, and time-evolution figures.


## Validation pass

The validation pass established that:

- the epoch-95 single-Re checkpoint strict-loads with 20,778,018 state
  elements;
- its model-only warm start introduces exactly 390 expected gated-adapter and
  FiLM keys;
- the epoch-55 multi-Re checkpoint strict-loads with 22,291,436 state
  elements and records denormalized validation relative L2
  0.025518675602041185;
- all ten real arrays have shape (1000,251,128,128,4);
- public loader splits are 8000/1000/1000, every balanced training batch
  contains all ten regimes, and sub_t=5 spans 51 frames from t=0 to t=5;
- both new KH validators and both existing DT four-channel validators pass;
- direct assertions verify time-local loss semantics and deterministic tiny
  validation sampling; and
- Python syntax and Git whitespace checks pass.

The established Apptainer image does not include pytest, so the committed
pytest files were mirrored by direct assertions in that same runtime. All
eight public scOT recipe guards passed after the shared trainer extension.
