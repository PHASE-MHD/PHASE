# PHASE

PHASE is a physics-adapted neural-operator framework for two-dimensional
incompressible magnetohydrodynamics. It combines POSEIDON transfer learning,
Reynolds-number conditioning, Helmholtz projection, physics-informed losses,
and residual diffusion for decaying turbulence and Kelvin--Helmholtz
instability.

## Install

PHASE requires Python 3.11. Create the pinned environment and install the
package from the repository root:

```bash
conda env create -f environments/phase-cuda121.yml
conda activate phase
python -m pip install -e .
```
## Model artifacts

For anonymous review, permanent project links and pretrained model artifacts
are withheld (`<ANONYMIZED_MODEL_ARTIFACT_URL>`). They will be disclosed after
the review process. The training
and evaluation workflows remain fully specified by the included configurations
and documentation.

## Workflows

- Generate MHD trajectories with [MHD_DataGeneration](MHD_DataGeneration/README.md).
- Convert data and fit training-only statistics using the
  [preprocessing guide](docs/preprocessing.md).
- Reproduce the tFNO and DINO baselines, model ablations, decaying-turbulence
  PHASE models, and Kelvin--Helmholtz PHASE models through
  [docs/README.md](docs/README.md).
- Run held-out-test [evaluation](docs/evaluation.md) and
  [visualization](docs/visualization.md).

Canonical YAML files under [configs](configs/README.md) define every model,
dataset split, normalization, loss, optimizer, warm start, and checkpoint
selection rule.

For example:

```bash
export DATA_ROOT=/path/to/prepared/data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/turbulence/single_re/scot_re1000.yaml
```

Training data and model checkpoints are external artifacts and are not tracked
in Git.

## Development check

```bash
python -m compileall -q src scripts
python -m pip install -e ".[dev]"
python -m build
```

PHASE is released under the [MIT License](LICENSE). Adapted third-party
components are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
