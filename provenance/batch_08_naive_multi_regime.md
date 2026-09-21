# Batch 8: naive multi-regime conditioning

Batch 8 adds the exact naive multi-regime three-channel scOT ablation. It
extends the audited Batch 7 model by appending constant standardized
`log10(Re)` and `log10(Rm)` channels at the input patch embedding.

Implemented items:

- ten locked `Re=Rm` regimes from 80 through 4500;
- independent seeded 800/100/100 splits per regime;
- balanced batches containing one sample from every regime;
- model-only warm start from the corrected Batch 7 single-Re checkpoint;
- zero-initialized Re/Rm channel weights;
- separate effective learning rates for copied and new input weights;
- per-sample viscosity and resistivity in the MHD residual loss;
- recipe guards, sampler tests, and strict checkpoint-compatibility checks;
- path-independent configuration and public training documentation.

The public trainer does not copy TensorBoard, prediction-plot, or spectrum
side effects from the legacy trainer. These did not affect optimization or
checkpoint selection. The historical launcher did not fully seed PyTorch, so
a fresh optimization trajectory is not expected to be bitwise identical even
though the activated computation and checkpoint format are preserved.

## Validation pass

The consolidation audit verified that:

- public and legacy split membership, normalized samples, metadata, and balanced
  sampler batches are bit-identical on synthetic ten-regime data;
- the public and legacy models strictly restore the same epoch-59 state and
  produce bit-identical conditioning maps and forward predictions;
- the corrected epoch-98 single-Re checkpoint expands from five to seven input
  channels while leaving both new channels exactly zero initialized;
- the public model has 20,777,742 parameters, and the final checkpoint restores
  844 AdamW states in groups of 843 and 1 tensors;
- copied patch-input channels receive a `1e-4` gradient multiplier, preserving
  effective learning rates of `1e-7` and `1e-3` for copied and new slices;
- public and legacy per-sample MHD objectives and prediction gradients are
  bit-identical; and
- the existing Batch 6 and Batch 7 recipe guards still pass.

The historical job targeted 100 epochs but its surviving log contains complete
summaries only through epoch 61, followed by a partial epoch 62. The reported
epoch-59 checkpoint is byte-identical to the final mutable best-checkpoint file;
there is therefore no ambiguity in the weights used for the ablation result.

The established Apptainer image does not include pytest. The pytest suites are
committed, and equivalent direct assertions plus full source compilation were
run successfully in that image.

## Current paper artifact update

The current ablation table supersedes the original epoch-59 snapshot with the
continuation checkpoint at local epoch 53 (effective epoch 113). The canonical
YAML remains the clean 100-epoch recipe; current artifact identity and metrics
are recorded in `ablations/naive_multi_regime.md` and
`checkpoint_manifest.yaml`.
