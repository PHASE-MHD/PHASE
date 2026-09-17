# Reproducible environments

Exact Python 3.11 environment files for the supported Gadi GPU platforms will
be added after dependency reconciliation. Portable dependency ranges belong in
`pyproject.toml`; CUDA-specific pins belong here.

The default Python executable on the Gadi login node is currently Python
3.7.7 and is not supported by PHASE. Installation, tests, and training must be
run inside one of the documented Python 3.11 environments or containers.
