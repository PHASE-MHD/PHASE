# Reproducible environments

PHASE supports Python 3.11. The Gadi login-node Python is not supported.

Create the pinned CUDA 12.1 environment from the repository root:

    conda env create -f environments/phase-cuda121.yml
    conda activate phase
    python -m pip install -e .

The environment pins direct dependencies used by the audited paths. Record the
realized environment for every production run:

    conda list --explicit > "${OUTPUT_ROOT}/environment-explicit.txt"
    python -m pip freeze > "${OUTPUT_ROOT}/pip-freeze.txt"

scOT is deliberately installed separately. Use the audited no-dependency
command in README_poseidon.md so upstream metadata cannot replace PyTorch.

Before requesting a GPU, run:

    phase-validate-configs configs
    pytest -q -m "not gpu and not checkpoint"

The legacy consolidation container uses Python 3.10 and is provenance only,
not the public release environment.
