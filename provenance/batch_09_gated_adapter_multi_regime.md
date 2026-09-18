# Batch 9: gated-adapter multi-regime conditioning

Batch 9 adds the exact three-channel gated-adapter multi-regime scOT ablation.
It replaces constant Re/Rm input maps with conditioning applied through every
scOT encoder/decoder block and at the output.

Implemented items:

- ten locked `Re=Rm` regimes from 80 through 4500;
- independent seeded 800/100/100 splits and balanced ten-regime batches;
- model-only warm start from the corrected Batch 7 single-Re checkpoint;
- 32 zero-initialized, channel-gated residual adapters;
- zero-initialized output FiLM conditioning;
- separate optimizer groups for copied and conditioning parameters;
- per-sample viscosity and resistivity in the MHD residual loss;
- strict recipe guards and checkpoint-compatibility tests; and
- path-independent configuration and public training documentation.

The public trainer omits legacy TensorBoard, prediction-plot, and spectrum side
effects because they do not affect optimization or checkpoint selection. The
historical launcher did not fully seed PyTorch, so a fresh run is not expected
to be bitwise identical even though the active computation and checkpoint
format are preserved.

## Validation pass

The consolidation audit verified that:

- the public config preserves the legacy architecture, normalization, ten
  regimes, balanced effective batch of 10, per-sample transport coefficients,
  loss weights, learning rates, weight decay, and validation-loss selection;
- all Batch 6--9 recipe guards pass, while deliberate changes to conditioning,
  adapter placement, normalization, PDE weight, checkpoint metric, magnetic
  initialization, and temporal subsampling are rejected;
- the public model installs 32 adapters and declares exactly 390 new warm-start
  state keys: 384 adapter entries and six output-FiLM entries;
- the corrected epoch-98 single-Re checkpoint loads with no missing keys beyond
  that exact allowlist and no unexpected keys;
- before conditioning training, predictions at `Re=80` and `Re=4500` are
  bit-for-bit identical to the single-Re warm-start prediction;
- the historical epoch-44 model loads strictly with 1,234 state tensors and
  22,289,366 total state elements;
- its AdamW state restores 844 warm-started tensors and 324 conditioning
  tensors, 1,168 optimizer states in total, and scheduler `last_epoch=45`; and
- repository-wide source, tests, and scripts compile in the established
  Apptainer environment.

The source comparison against the canonical legacy model found only package
imports/registration, the explicit warm-start allowlist, and removal of
inactive options from the locked three-channel factory. The legacy final
Helmholtz call was a no-op for this config; omitting it does not alter the
active computation.

The established Apptainer image does not include pytest. The pytest suites are
committed, and equivalent direct assertions were run successfully. The image
uses Python 3.10.13; the public package remains declared for Python 3.11.

The historical job targeted 100 epochs, but the surviving log contains
complete summaries only through epoch 50 and no normal-completion marker. The
reported epoch-44 checkpoint is byte-identical to the mutable best-checkpoint
file, so its provenance is unambiguous without claiming that the schedule
finished.
