# DT single-Re residual diffusion provenance

This record identifies the canonical corrected single-Re PHASE diffusion run
at `Re=Rm=1000`. Unlike the historical warm-start artifact, this model learns
residuals for all four fields `[ux,uy,Bx,By]` from random initialization.

- public recipe: `configs/turbulence/single_re/phase_re1000.yaml`;
- legacy training job: `179342063.gadi-pbs`;
- completed epochs: 0--99;
- checkpoint selection: denormalized relative L2 at scheduled validation;
- selected epoch: 90;
- selected denormalized relative L2: `0.028242717292159797`;
- selected denormalized MSE: `1.9426884546192013e-05`;
- checkpoint state entries: 349;
- checkpoint byte size: `5144073110`;
- SHA-256: `ff57dd73a0ba31a43f65acdda6cf0bb188128a71700cf39204b22bc7fcdf998e`.

The run uses paired four-channel normalization, 32-step EDM sampling, residual
targets, and full-field Helmholtz projection after physical reconstruction.
The checkpoint is external and located through
`PHASE_SINGLE_RE_DIFFUSION_CHECKPOINT` or the canonical artifact manifest.
