# Command-line scripts

- `prepare_data.py`: convert Dedalus HDF5 trajectories or vector-potential
  arrays to canonical PHASE `.npy` layouts.
- `compute_statistics.py`: compute train-only trajectory or diffusion
  normalization statistics.
- `train_tfno.py`: train the previous-study, no-warm-start tFNO baseline.
- `verify_artifact.py`: check an external checkpoint against its documented
  SHA-256 digest.
- `generate_diffusion_features.py`: generate full-field DINO conditioner/DNS
  pairs from the released tFNO.
- `train_dino.py`: train the previous-study full-field EDM diffusion baseline.
- `train_scot.py`: train the guarded three-channel scOT ablations. Batches 6-9
  implement the no-transfer, POSEIDON-transfer, naive multi-regime, and gated-
  adapter multi-regime recipes.

`generate_scot_diffusion_features.py` exports physical-unit scOT/DNS pairs for
single- or multi-Re residual diffusion. `train_dino.py` now dispatches both the
locked previous DINO full-field recipe and the guarded PHASE residual recipes.
Evaluation and visualization entry points will be added in later audited batches.
