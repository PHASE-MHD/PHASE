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

## Paranoid validation pass

A second end-to-end audit covered every activated layer of the recipe:

- every scientific and training field canonicalizes exactly to the corrected
  legacy YAML after accounting for renamed path/loader keys;
- seed-42 train, validation, and test membership, coordinate grids,
  subsampling, physics normalization, samples, and denormalization are exact;
- fresh POSEIDON expansion produces the same 844 state keys and bit-identical
  initial tensors when resolved from the same cached upstream snapshot;
- strict restoration of the epoch-98 model, both AdamW parameter groups, all
  844 optimizer states, and the dummy scheduler succeeds without translation;
- a synthetic full training epoch gives bit-identical loss components,
  gradient-clipped parameter updates, and final parameters in the public and
  legacy loops;
- normalized validation loss and denormalized relative-L2/MSE metrics are
  bit-identical between the public and legacy validation paths;
- the public and legacy checkpoint dictionaries have identical keys and
  values for model, optimizer, scheduler, epoch, selection loss, and
  denormalized metrics;
- the public CLI imports successfully, all Python sources compile in the
  project container, and no data arrays or checkpoints are tracked;
- the shared Batch 6 path still passes its recipe guard and strictly restores
  its epoch-95 checkpoint with 11,212,890 parameters.

The public validator evaluates the normalized objective and denormalized
metrics in two deterministic passes, whereas the legacy validator computed
both in one pass. This changes evaluation overhead only; the audited values
are identical. The public trainer also omits TensorBoard and legacy prediction
plot/spectrum side effects. Those outputs do not enter the objective or
checkpoint criterion. Because the legacy launcher did not seed PyTorch's
shuffled training order, neither implementation can reproduce the historical
optimization trajectory bit-for-bit from the surviving configuration alone.

The test container lacks the `pytest` package. Equivalent direct assertions
were run successfully inside the established Apptainer environment, and the
pytest test files are included for standard development environments.
