# Batch 7: scOT with POSEIDON transfer learning

Batch 7 adds the exact three-channel transfer-learning ablation while reusing
the audited Batch 6 data, model, loss, optimizer, and trainer modules.

Implemented items:

- locked `Re=Rm=1000`, batch-size-16 configuration;
- pretrained `camlab-ethz/Poseidon-T` initialization;
- native POSEIDON velocity normalization;
- mean-velocity magnetic-input initialization and zero magnetic-output
  initialization;
- magnetic residual learning;
- separate copied/new-channel AdamW update rates;
- recipe-specific guards preventing accidental scratch/TL config mixing;
- opt-in parity for the later legacy loss-component logging schema;
- unit checks and strict epoch-98 checkpoint compatibility test;
- setup, training, and provenance documentation.

## Validation pass

The audit verified:

- the activated scientific and optimization values match the corrected legacy
  YAML, including `nu=eta=1e-3`;
- the public and legacy model wrappers strictly load the same checkpoint and
  return bit-identical predictions;
- the public model parameter count is 20,776,206;
- optimizer group tensor counts, learning rates, and weight decays exactly
  match the saved optimizer state;
- total objective, prediction gradients, and all 19 logged loss components are
  bit-identical;
- checkpoint selection remains based on normalized validation loss;
- Batch 6 remains guarded as batch size 1 without pretrained weights.

The test container lacks the `pytest` package. Equivalent direct assertions
were run successfully inside the established Apptainer environment, and the
pytest test files are included for standard development environments.
