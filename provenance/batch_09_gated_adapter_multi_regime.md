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

The consolidation audit checks that the locked config, warm-start contract,
zero-correction initialization, 32-block placement, optimizer grouping,
checkpoint metadata, and `log10(Re/Rm)` standardization match the historical
artifact. The final validation results and any limitations of the available
environment are recorded when the audit is completed.
