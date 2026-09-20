# Batch 16: artifact and GPU acceptance

Batch 16 closes the gap between synthetic release tests and the external
artifacts used for the reported tFNO, DINO, scOT, and PHASE model chains.

## Canonical checkpoint manifest

`checkpoint_manifest.yaml` records a stable artifact ID, purpose, environment
variable, repository-relative legacy path, byte size, SHA-256 digest, and
selected epoch for every audited checkpoint. Checkpoints remain external and
are not committed to the public repository. The verifier resolves either the
per-artifact environment variable or `PHASE_ARTIFACT_ROOT` and rejects missing,
size-mismatched, or hash-mismatched files.

All manifest digests were recomputed from the legacy files on 2026-09-19. This
audit caught and excluded a similarly named multi-Re DT diffusion checkpoint:
the canonical reported epoch-95 artifact is under the `resume_epoch45` chain
and has SHA-256
`97c24e526430889c173472770d97389531885599696dad6d1c514dc6094fd1b4`.

## Strict checkpoint loading

Final PBS job `179375948.gadi-pbs` resolved all 16 manifest paths, checked
their byte sizes, and ran the complete marked checkpoint suite in the project
container. Result: `12 passed` in 311.99 seconds, exit status 0. The tests cover
the previous tFNO/DINO artifacts, three-channel scOT ablations, four-channel DT
scOT, corrected SR and canonical MR DT PHASE, and canonical t=[0,5] KH
scOT/PHASE chains.

## Real-GPU acceptance

PBS job `179375235.gadi-pbs` ran the four public model families on a V100.
Result: `4 passed` in 16.58 seconds, exit status 0. Each test constructs the
public model, performs a CUDA forward and backward pass, and checks finite
outputs and gradients. The diffusion cases also run the public sampler.

The residual sampler intentionally returns the normalized residual. Production
inference denormalizes that residual, reconstructs the physical full field as
condition plus residual, and only then applies Helmholtz projection separately
to `(ux,uy)` and `(Bx,By)`. The GPU acceptance test follows this contract and
verifies the projected full-field divergence. An earlier draft test projected
the raw sampler output directly; that was a test error, not a runtime defect.

## Additional validation

- full ordinary CPU suite: `152 passed, 16 deselected`;
- manifest unit tests cover uniqueness, portable relative paths, size checks,
  and SHA-256 rejection;
- the Python 3.11 wheel built successfully and its 84 files contained no model
  artifacts, arrays, logs, generated outputs, caches, or bytecode;
- all 15 artifacts present at the start of this audit passed byte-size checks;
- all 15 initial SHA-256 digests were independently recomputed;
- no checkpoint, generated output, or legacy repository file is committed.

The corrected single-Re DT residual PHASE job `179342063.gadi-pbs` subsequently
completed epochs 0--99 with exit status 0. Its selected epoch-90 checkpoint was
then frozen as the sixteenth manifest artifact, independently hashed, and
strict-loaded through the public single-Re residual recipe. It records
denormalized relative L2 `0.028242717292159797` and SHA-256
`ff57dd73a0ba31a43f65acdda6cf0bb188128a71700cf39204b22bc7fcdf998e`.

## Paranoid audit follow-up

The 2026-09-20 follow-up audit independently recomputed all 16 SHA-256
digests, reran all 12 external-checkpoint compatibility tests, validated all 15
public recipes, and reran the complete ordinary suite. Results were `12 passed`
for checkpoint compatibility and `153 passed, 16 deselected` for the ordinary
suite after hardening.

The audit found no model, checkpoint, normalization, warm-start, residual
reconstruction, or Helmholtz-projection defect. It did find that the manifest
loader trusted individual record structure too heavily. The loader and unit
tests now reject malformed records, duplicate IDs, environment variables and
paths, invalid sizes, hashes and epochs, absolute paths, and parent traversal.

The real-CUDA source path was unchanged by this hardening. Its retained V100
acceptance remains `4 passed`; this CPU-side follow-up did not resubmit that
already successful GPU job. End-to-end data-to-evaluation smoke tests remain
the explicit pre-release TODO in `docs/release_checklist.md`.
