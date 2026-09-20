# Batch 18: reduced end-to-end KH acceptance

Batch 18 adds a six-stage, data-to-model acceptance chain for the canonical
Kelvin-Helmholtz `t=[0,5]` recipes. It mirrors the successful Batch 17 DT chain
while preserving the KH-specific 51-frame contract, normalization, losses,
warm starts, and residual diffusion behavior.

The acceptance data sizes are 160/10/10 trajectories for single Re and
1,600/100/100 trajectories for multi Re. All training stages use ten epochs.
Generated features and normalization statistics come only from this reduced
chain; no production checkpoint is injected.

Files added:

- `configs/acceptance/kh_20pct_10ep/`
- `scripts/submit_kh_acceptance_chain.sh`
- `docs/kh_acceptance_chain.md`

Static and dry-run validation results, scheduler job IDs, final epochs, and
held-out smoke metrics should be appended here after execution.

## Pre-submission validation

- all four configs passed static validation;
- a strict recursive diff confirmed that every departure from the canonical KH recipes is an allowed acceptance-only sample, epoch, monitoring, or path override;
- the internal single-to-multi-Re warm-start chain passed direct assertions;
- all ten KH source arrays were found;
- source and generated PBS scripts passed shell syntax checks;
- the supported project container compiled the source, scripts, and focused acceptance test; and
- `git diff --check` passed.

## Scheduler record

Submitted on 2026-09-21 as one strict `afterok` chain:

- single-Re scOT: `179451820.gadi-pbs`, `gpuhopper`, 6 hours;
- single-Re feature/statistics: `179451821.gadi-pbs`, `gpuvolta`, 2 hours;
- single-Re residual diffusion: `179451822.gadi-pbs`, `gpuhopper`, 2 hours;
- multi-Re scOT: `179451823.gadi-pbs`, `gpuhopper`, 8 hours;
- multi-Re feature/statistics: `179451824.gadi-pbs`, `gpuvolta`, 2 hours; and
- multi-Re residual diffusion: `179451825.gadi-pbs`, `gpuhopper`, 10 hours.

At submission, the first stage was queued and all downstream stages were held by the expected dependencies.

The original long-walltime chain was canceled before execution and replaced by
jobs `179452368`--`179452373` with shorter limits. KH acceptance is an optional
integration sanity check and is not a public-release gate: the canonical
`t=[0,5]` KH artifacts already passed strict checkpoint compatibility in Batch
16. Final outcomes may be appended if the replacement chain is allowed to run.
