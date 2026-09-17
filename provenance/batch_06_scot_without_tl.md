# Batch 6: scOT without transfer learning

Batch 6 adds the first scOT ablation to the consolidated runtime. The wrapper
preserves the legacy parameter names so reported checkpoints remain strictly
loadable, while model construction is handled by the shared PHASE factory.

Implemented items:

- three-channel scOT wrapper for `[u_x,u_y,A]`;
- random initialization with no POSEIDON transfer weights;
- residual prediction for velocity and vector potential;
- opt-in spectral `B=curl(A)` training loss and component logging;
- guarded batch-size-1 trainer and portable configuration;
- unit tests and strict legacy-checkpoint compatibility test;
- setup, training, and provenance documentation.

The strict compatibility check loaded the reported epoch-95 checkpoint with no
missing or unexpected keys and confirmed 11,212,890 model parameters.

## Paranoid validation pass

The post-implementation audit verified:

- every scientific and optimization config value matches the locked legacy
  YAML; only portable paths, explicit coordinate ranges, and key names differ;
- the public model and legacy model load the epoch-95 state strictly and give
  bit-identical output for the same input (`max_abs_diff=0`);
- seed-42 train/validation/test indices, coordinate grids, normalized inputs,
  and targets are bit-identical to the legacy dataset implementation;
- the complete objective and all 17 logged components are bit-identical;
- gradient clipping plus AdamW produces bit-identical parameter updates;
- normalized validation loss, denormalized relative L2, and denormalized MSE
  are numerically identical to the legacy validator;
- the checkpoint optimizer contains one parameter group with learning rate
  `5e-4`, weight decay `0`, and betas `[0.9,0.999]`; its dummy scheduler keeps
  that learning rate unchanged.

The audit restored one subtle legacy behavior: the original trainer consumed
one shuffled training iterator for a shape probe before constructing the
randomly initialized model. The public trainer now preserves that RNG
consumption order and enables cuDNN benchmarking in the same place.

## Reproducibility limitation inherited from the legacy run

Seed 42 fixes the dataset permutation, but the legacy training entry point did
not set or record a PyTorch model-initialization seed. Therefore, the public
code exactly reproduces the architecture, data split, objective, optimizer,
training order, and checkpoint semantics, but a fresh run is not guaranteed to
recreate the epoch-95 weights bit-for-bit. The immutable checkpoint hash and
strict-load/numerical-parity tests validate the reported trained artifact.
