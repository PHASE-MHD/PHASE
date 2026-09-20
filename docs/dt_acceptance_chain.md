# Decaying-turbulence end-to-end acceptance chain

This acceptance experiment checks the complete PHASE training path with reduced
data and training time. It is an integration test, not a replacement for the
100-epoch paper results.

## Scope

The chain runs these stages in order:

1. single-Re Re=Rm=1000 four-channel scOT;
2. single-Re scOT feature generation and train-only paired statistics;
3. single-Re four-channel residual diffusion from random initialization;
4. ten-regime multi-Re scOT, warm-started from stage 1;
5. multi-Re scOT feature generation and train-only per-Re paired statistics;
6. multi-Re residual diffusion, warm-started model-weights-only from stage 3.

Every training stage runs for 10 epochs. The canonical seed-42 split is formed
first and is then deterministically reduced, so no trajectory crosses split
boundaries. Training uses 20% of the canonical training split. To bound
acceptance-test cost, validation and test each use 10% of their canonical
held-out split.

The resulting trajectory counts are:

- single Re: 160 train, 10 validation, and 10 test;
- multi Re: 160 train, 10 validation, and 10 test per Re;
- multi-Re totals: 1,600 train, 100 validation, and 100 test.

The reduced validation/test sets are suitable only for acceptance checks. They
must not be used as paper metrics.

## Locked behavior

The acceptance configs copy the production architecture, loss, optimizer,
normalization, batching, residual-learning, and Helmholtz-projection settings.
Only the sample fractions, epoch count, validation interval, artifact paths,
and explicit acceptance flag differ.

Both diffusion stages learn residuals for all four fields
`[ux, uy, Bx, By]`. Projection is applied separately to the velocity and
magnetic pairs after reconstructing `scOT + residual`, and the projected field
is converted back to residual form for the EDM objective.

The multi-Re scOT uses balanced batches containing all ten regimes. Multi-Re
diffusion uses train-only per-Re paired min-max statistics and does not apply a
second sample reduction to the already reduced feature stores.

## Configs

- `configs/acceptance/dt_20pct_10ep/single_re_scot.yaml`
- `configs/acceptance/dt_20pct_10ep/single_re_phase.yaml`
- `configs/acceptance/dt_20pct_10ep/multi_re_scot.yaml`
- `configs/acceptance/dt_20pct_10ep/multi_re_phase.yaml`

## Run on Gadi

The launcher creates six PBS scripts under the ignored run directory and
submits them with `afterok` dependencies:

```bash
cd /path/to/PHASE
bash scripts/submit_dt_acceptance_chain.sh
```

Machine-specific roots can be overridden:

```bash
DATA_ROOT=/path/to/mhd/data \
POSEIDON_ROOT=/path/to/poseidon \
POSEIDON_DEPS=/path/to/poseidon/dependencies \
MHD_PINO_DEPS=/path/to/mhd/dependencies \
CONTAINER=/path/to/phase.sif \
PROJECT=project-code \
RUN_ROOT=/path/to/output \
QUEUE=gpuhopper \
bash scripts/submit_dt_acceptance_chain.sh
```

All variables above except `RUN_ROOT` and `QUEUE` are required; the launcher
does not contain user- or site-specific filesystem defaults.

The default acceptance limits are:

- single-Re scOT: `gpuhopper`, 8 hours;
- single-Re feature/statistics: `gpuvolta`, 2 hours;
- single-Re residual diffusion: `gpuhopper`, 1 hour;
- multi-Re scOT: `gpuhopper`, 4 hours;
- multi-Re feature/statistics: `gpuvolta`, 2 hours;
- multi-Re residual diffusion: `gpuhopper`, 6 hours.

The launcher prints and records every job ID in
`$RUN_ROOT/submitted_jobs.txt`. Logs are written to `$RUN_ROOT/logs/`;
checkpoints to `$RUN_ROOT/training/checkpoints/dt_20pct_10ep/`; generated
features to `$RUN_ROOT/features/acceptance/dt_20pct_10ep/`; and statistics to
`$RUN_ROOT/stats/acceptance/dt_20pct_10ep/`.

A failed stage prevents all downstream jobs from starting. Rerunning into an
existing run root is rejected to prevent accidental artifact mixing.
