# Reproducing PHASE experiments

This guide starts from simulation arrays and ends with held-out-test reports.
DNS data, POSEIDON weights, generated features, and checkpoints are external.

## Install and preflight

Create the environment in environments/README.md, install scOT as described in
README_poseidon.md, and define:

    export DATA_ROOT=/path/to/processed/data
    export STATS_ROOT=/path/to/train_only/statistics
    export FEATURE_ROOT=/path/to/generated/features
    export CHECKPOINT_ROOT=/path/to/external/checkpoints
    export OUTPUT_ROOT=/path/to/phase/outputs
    mkdir -p "${STATS_ROOT}" "${FEATURE_ROOT}" "${OUTPUT_ROOT}/checkpoints"

Validate semantics first and local paths second:

    phase-validate-configs configs
    phase-validate-configs --check-paths configs/path/to/recipe.yaml

Static validation loads neither models nor data. Path checks require expanded
environment variables and existing inputs.

## Data

Follow docs/data_format.md and docs/preprocessing.md. Fit statistics on the
training split only; never fit on validation or test samples. The split seed
is 42.

## Eight DT ablations

| Row | Recipe | Prerequisite |
| --- | --- | --- |
| tFNO | configs/previous_baseline/tfno/re1000.yaml | Re=1000 vector-potential data and train statistics |
| DINO | configs/previous_baseline/dino/re1000.yaml | released tFNO conditioner, full-field features and statistics |
| scOT without TL | configs/ablations/scot_without_tl/re1000.yaml | Re=1000 vector-potential data |
| scOT with TL | configs/ablations/scot_with_tl/re1000.yaml | Poseidon-T |
| naive MR | configs/ablations/naive_multi_regime/multi_re.yaml | corrected scOT-with-TL checkpoint |
| gated-adapter MR | configs/ablations/gated_adapter_multi_regime/multi_re.yaml | corrected scOT-with-TL checkpoint |
| four-channel HP/physics | configs/turbulence/multi_re/scot.yaml | configs/turbulence/single_re/scot_re1000.yaml |
| PHASE | configs/turbulence/multi_re/phase.yaml | MR scOT features and documented historical SR warm start |

Use scripts/train_tfno.py for tFNO, scripts/train_scot.py for deterministic
scOT, and scripts/train_dino.py for DINO or PHASE. Model-specific docs contain
feature-generation and statistics commands.

The corrected SR DT PHASE recipe is
configs/turbulence/single_re/phase_re1000.yaml. It trains four-channel
residual diffusion from random initialization. It is distinct from the
historical full-field warm start used by the reported MR PHASE run.

## KH

The canonical KH interval is t=[0,5]. Train in order:

1. configs/kh/single_re/scot_re1000.yaml
2. configs/kh/multi_re/scot_t0_5.yaml
3. configs/kh/single_re/phase_re1000.yaml
4. configs/kh/multi_re/phase.yaml

Multi-Re scOT and diffusion use model-only single-Re warm starts. Both
diffusion recipes learn residuals in all four fields and project the
reconstructed full velocity and magnetic fields.

## Evaluation

Paper metrics must use the held-out test split:

    python scripts/evaluate_error.py --help
    python scripts/visualize.py --help

Both require explicit Re. Reports record split metadata, hashes, sample IDs,
and model family. Never relabel validation diagnostics as test results.

For each production run archive the expanded config, Git commit, environment,
checkpoint hash, selected epoch and metric, split, Re/Rm, seed, and command.
