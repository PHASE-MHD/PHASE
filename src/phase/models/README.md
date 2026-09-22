# Models

Deterministic neural operators used by the baseline ablations and PHASE.

- `tfno.py`: three-dimensional tensorized Fourier neural operator used by the
  tFNO baseline and as the DINO conditioner.
- `layers/spectral_layers.py`: factorized three-dimensional Fourier convolution.
- `scot_mhd.py`: POSEIDON/scOT extension for three-channel `[ux,uy,A]` and
  four-channel `[ux,uy,Bx,By]` MHD prediction, residual learning, and Helmholtz projection.
- `scot_mhd_naive_re.py`: naive multi-regime conditioning through constant
  standardized `log10(Re)` and `log10(Rm)` input maps.
- `scot_mhd_gated_re.py`: gated-adapter and output-FiLM conditioning for the
  multi-regime scOT models.
- `checkpoint_mapping.py`: conversion of released tFNO checkpoint
  keys to the local tFNO layout.
- `model_factory.py`: config-driven deterministic model construction.
- `__init__.py` and `layers/__init__.py`: public package exports.
