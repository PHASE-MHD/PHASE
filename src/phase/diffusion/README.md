# Diffusion

Conditional EDM diffusion used by DINO and PHASE.

- `models/elucidated_diffusion.py`: EDM training and sampling objective.
- `models/backbones/`: conditional U-Net denoiser.
- `models/components/`: attention, residual, resolution, and normalization layers.
- `models/helmholtz_projection.py`: divergence-free projection for velocity and magnetic pairs.
- `models/diffusion_factory.py`: config-driven model construction.
- `utils/`: diffusion tensor and model helpers.

The previous DINO baseline predicts the full trajectory conditioned on a tFNO
prediction. PHASE instead predicts `DNS - scOT` for all four physical fields
`[ux,uy,Bx,By]`. PHASE reconstructs `scOT + residual` in physical units and
applies Helmholtz projection to the complete velocity and magnetic fields; the
residual alone is not projected.

The diffusion U-Net is not directly Re/Rm conditioned. Multi-regime behavior
comes from the conditioned scOT trajectory and per-regime normalization
metadata. See `docs/turbulence_residual_diffusion.md` and
`docs/kh_residual_diffusion.md` for complete workflows.
