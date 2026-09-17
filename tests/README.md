# Tests

- `unit/`: fast CPU tests for isolated behavior.
- `integration/`: multi-component CPU/GPU smoke tests.
- `checkpoint_compatibility/`: tests requiring external legacy checkpoints.
- `fixtures/`: small synthetic test data only.

GPU and checkpoint tests must use their corresponding pytest markers.
