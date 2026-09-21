# Reproducible environments

PHASE supports Python 3.11.

Create the pinned CUDA 12.1 environment from the repository root:
    conda env create -f environments/phase-cuda121.yml
    conda activate phase
    python -m pip install -e .

Record the realized environment for every production run:

    conda list --explicit > "${OUTPUT_ROOT}/environment-explicit.txt"
    python -m pip freeze > "${OUTPUT_ROOT}/pip-freeze.txt"

scOT is installed separately. Use the audited no-dependency
command in README_poseidon.md so upstream metadata cannot replace PyTorch.
