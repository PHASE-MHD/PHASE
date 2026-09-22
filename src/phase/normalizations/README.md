# Normalization

normalization strategies used by PHASE:
- `physics_norm.py`: fixed channel-wise scaling for scOT models. Shared scales for `(u_x, u_y)` and `(B_x, B_y)` preserve compatibility with Helmholtz projection.
- `min_max_norm.py`: min-max normalization for the tFNO/DINO baselines and paired min-max normalization for residual diffusion. The multi-regime variant uses per-Re statistics and log-Re interpolation for unseen Reynolds numbers.
- `identity_norm.py`: pass-through normalization.
- `standard_norm.py`: optional mean-standard-deviation normalization.
- `normalization_factory.py`: configuration-based construction of normalization modules.

Diffusion models use separate statistics for conditioning inputs and residual targets. Multi-regime diffusion uses statistics computed independently for each training Reynolds number.