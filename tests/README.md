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
reported epoch-59 naive multi-regime checkpoint before running the marked
checkpoint test.
