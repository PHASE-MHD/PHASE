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
