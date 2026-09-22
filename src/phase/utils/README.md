# Utilities

- `config.py`: loads YAML configurations.
- `batches.py`: unpacks optional regime metadata and forwards Re/Rm conditioning.
- `data_utils.py`: identifies physical channels and applies dataset normalization transforms.
- `diffusion_tensor_normalization.py`: handles channel-aware conditioner and residual normalization, reconstructs full fields, and applies full-field Helmholtz projection for residual diffusion.
- `fourier_utils.py`: provides differentiable periodic Fourier derivatives, Laplacians, and centered time derivatives.