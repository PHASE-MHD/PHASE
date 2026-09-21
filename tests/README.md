# Tests

- `unit/`: fast CPU tests for isolated behavior.
- `integration/`: multi-component CPU/GPU smoke tests.
- `checkpoint_compatibility/`: tests requiring external legacy checkpoints.
- `fixtures/`: small synthetic test data only.

GPU and checkpoint tests must use their corresponding pytest markers.

The external tFNO compatibility test is enabled with:

```bash
export PHASE_LEGACY_TFNO_CHECKPOINT=/path/to/tFNO_Re1000_checkpoint.pt
pytest -m checkpoint tests/checkpoint_compatibility/test_tfno_legacy_checkpoint.py
```

The external previous-study DINO artifacts are checked with:

```bash
export PHASE_OFFICIAL_TFNO_CHECKPOINT=/path/to/tfno_Re1000.pt
export PHASE_LEGACY_DINO_CHECKPOINT=/path/to/dino_epoch100.pt
pytest -m checkpoint tests/checkpoint_compatibility/test_dino_legacy_checkpoints.py
```

The scOT-without-transfer checkpoint check is enabled with:

```bash
export PHASE_SCOT_WITHOUT_TL_CHECKPOINT=/path/to/scot_without_tl.pt
pytest -m checkpoint \
  tests/checkpoint_compatibility/test_scot_without_tl_legacy_checkpoint.py
```

The scOT transfer-learning checkpoint check is enabled with:

```bash
export PHASE_SCOT_WITH_TL_CHECKPOINT=/path/to/scot_with_tl.pt
pytest -m checkpoint \
  tests/checkpoint_compatibility/test_scot_with_tl_legacy_checkpoint.py
```

## Batch 8 checkpoint compatibility

Set both `PHASE_SCOT_WITH_TL_CHECKPOINT` and
`PHASE_NAIVE_MULTI_RE_CHECKPOINT` to the corrected single-Re warm start and
current paper-table continuation checkpoint (checkpoint-local epoch 53,
effective epoch 113) before running the marked checkpoint test.

## Batch 9 checkpoint compatibility

Set `PHASE_SCOT_WITH_TL_CHECKPOINT` to the corrected single-Re warm start and
`PHASE_GATED_ADAPTER_MULTI_RE_CHECKPOINT` to the rebased effective-epoch-90
checkpoint used by the current paper table, then run:

```bash
pytest -m checkpoint \
  tests/checkpoint_compatibility/test_gated_adapter_multi_regime_legacy_checkpoint.py
```

The test verifies strict final-checkpoint loading, model-only warm-start keys,
zero-correction initialization, exact warm-start prediction preservation,
optimizer grouping, and scheduler/checkpoint metadata.


## Batch 10 checkpoint compatibility

Set `PHASE_SINGLE_RE_FOUR_CHANNEL_CHECKPOINT` to the epoch-98 single-Re
four-channel checkpoint and `PHASE_MULTI_RE_FOUR_CHANNEL_CHECKPOINT` to the
reported epoch-91 multi-Re checkpoint, then run:

```bash
pytest -m checkpoint tests/checkpoint_compatibility/test_four_channel_scot_legacy_checkpoints.py
```

The test checks strict loading, model-only warm-start keys, checkpoint metadata,
parameter counts, and optimizer grouping. Unit tests additionally lock the two
recipes, direct-B transport metadata, Helmholtz divergence, mean-mode
preservation, and paired-normalization commutation.

## Batch 12 checkpoint compatibility

Set `PHASE_KH_SINGLE_RE_SCOT_CHECKPOINT` to the selected epoch-95 single-Re
KH checkpoint and `PHASE_KH_MULTI_RE_SCOT_CHECKPOINT` to the selected epoch-55
global-P99 t=[0,5] checkpoint, then run:

```bash
pytest -m checkpoint tests/checkpoint_compatibility/test_kh_scot_legacy_checkpoints.py
```

The test strict-loads both checkpoints and verifies the model-only warm-start
boundary. Unit tests lock the KH configs, reported loss-reduction semantics,
and the five-sample-per-regime tiny-validation subset.

For Batch 11 checkpoint compatibility, set
`PHASE_SINGLE_RE_DIFFUSION_CHECKPOINT` to the corrected epoch-90 residual
checkpoint, or set both
`PHASE_HISTORICAL_SR_DIFFUSION_CHECKPOINT` to the historical Re=1000
full-field diffusion warm start and `PHASE_MULTI_RE_DIFFUSION_CHECKPOINT` to
the reported epoch-95 MR PHASE checkpoint. The optional tests verify strict
architecture compatibility and recorded best metrics without committing large
checkpoint files.


## Batch 13 checkpoint compatibility

Set `PHASE_KH_SINGLE_RE_DIFFUSION_CHECKPOINT` to the selected epoch-95
single-Re KH residual checkpoint and `PHASE_KH_MULTI_RE_DIFFUSION_CHECKPOINT`
to the selected epoch-95 multi-Re t=[0,5] checkpoint, then run:

```bash
pytest -m checkpoint \
  tests/checkpoint_compatibility/test_kh_residual_diffusion_legacy_checkpoints.py
```

The test constructs the public U-Net on the meta device, strict-loads both large
checkpoints with mmap-backed tensors, and verifies their selected metrics and
349-key state schema. Unit tests lock the KH time contract, residual mode,
normalization, warm-start boundary, validation cadence, and full-state resume.

## Batch 14 evaluation

`unit/test_evaluation.py` verifies the common vector-potential/direct-B Fourier
derivatives, exact-field metrics, KH metric scope, duplicate sample protection,
model-family dispatch, and the test-only report contract.
The synthetic integration test in `integration/test_evaluation_inference.py`
exercises deterministic tFNO/scOT inference, full-field DINO sampling, and
residual PHASE reconstruction without requiring external checkpoints.

## Batch 16 artifact and GPU acceptance

Download checkpoints from their Hugging Face model cards, verify their SHA-256
digests against the published values, and set the checkpoint environment
variables named by `tests/checkpoint_compatibility`. Then run:

```bash
pytest -m checkpoint tests/checkpoint_compatibility
```

The real-GPU gate is:

```bash
pytest -m gpu tests/integration/test_gpu_acceptance.py
```

It performs CUDA forward/backward checks for tFNO, scOT, previous-DINO
full-field diffusion, and PHASE residual diffusion. The residual test mirrors
production inference: `sample()` returns a normalized residual; the runtime
then denormalizes it, adds it to the conditioner, and Helmholtz-projects the
reconstructed full velocity and magnetic fields.
