# Batch 17: reduced end-to-end DT acceptance

## Purpose

Batch 17 exercises the complete corrected decaying-turbulence PHASE chain from
raw four-channel trajectories through deterministic scOT training, diffusion
feature/statistics generation, residual-diffusion training, multi-Re warm
starts, and held-out evaluation.

This is an acceptance run, not a paper-result reproduction. Production
100-epoch configs and reported artifacts remain unchanged.

## Acceptance scope

- problem: 2-D incompressible MHD decaying turbulence;
- fields: `[ux, uy, Bx, By]`;
- regimes: single `Re=Rm=1000`, then all ten canonical `Re=Rm` values;
- split seed: 42;
- canonical split: 800 train, 100 validation, 100 test per Re;
- acceptance subset: 20% train, 10% validation, 10% test;
- epochs: 10 for each of the four training stages;
- scOT checkpoint metric: normalized validation loss;
- diffusion checkpoint metric: denormalized relative L2;
- diffusion target: four-field residual;
- diffusion projection: full reconstructed field, velocity and magnetic pairs
  projected separately.

Fractions are applied only after the canonical split is selected. The
single-Re residual model starts from random diffusion weights. The multi-Re
scOT warm-starts from the newly trained single-Re scOT checkpoint. The
multi-Re residual diffusion model warm-starts model weights only from the
newly trained single-Re residual checkpoint and resets epoch, optimizer, and
scheduler state.

## Files

Implementation and validation:

- `src/phase/data/neurops_dataset.py`
- `src/phase/data/multi_re_neurops_dataset.py`
- `src/phase/training/scot_trainer.py`
- `src/phase/training/dino_trainer.py`
- `src/phase/config_validation.py`
- `scripts/generate_scot_diffusion_features.py`

Acceptance recipes:

- `configs/acceptance/dt_20pct_10ep/single_re_scot.yaml`
- `configs/acceptance/dt_20pct_10ep/single_re_phase.yaml`
- `configs/acceptance/dt_20pct_10ep/multi_re_scot.yaml`
- `configs/acceptance/dt_20pct_10ep/multi_re_phase.yaml`

Launcher and instructions:

- `scripts/submit_dt_acceptance_chain.sh`
- `docs/dt_acceptance_chain.md`

## Pre-submission validation

- Python syntax compilation: passed.
- whitespace/error check with `git diff --check`: passed.
- focused unit/config suite: 62 passed.
- all public configuration files: 19 passed static validation.
- reduced single-Re split boundary regression: passed.
- reduced per-Re split regression: passed.
- clean residual-diffusion warm-start chain regression: passed.

## Scheduler record

The launcher writes exact job IDs to
`outputs/dt_20pct_10ep/submitted_jobs.txt`. Runtime outcomes, selected
checkpoint epochs, and held-out acceptance metrics must be appended here after
the dependent chain completes.

Submitted on 2026-09-20 to `gpuhopper`:

- single-Re scOT: `179408456.gadi-pbs`;
- single-Re feature generation/statistics: `179408457.gadi-pbs`;
- single-Re residual diffusion: `179408458.gadi-pbs`;
- multi-Re scOT: `179408459.gadi-pbs`;
- multi-Re feature generation/per-Re statistics: `179408460.gadi-pbs`;
- multi-Re residual diffusion: `179408461.gadi-pbs`.

At submission, the first job was queued and all downstream jobs were held by
the expected `afterok` dependencies.

The single-Re scOT completed successfully in 7 minutes with exit status zero.
The remaining five pending jobs were then replaced with shorter acceptance
limits and feature preparation on `gpuvolta`:

- single-Re feature generation/statistics: `179412378.gadi-pbs`,
  `gpuvolta`, 2 hours;
- single-Re residual diffusion: `179412379.gadi-pbs`, `gpuhopper`, 1 hour;
- multi-Re scOT: `179412383.gadi-pbs`, `gpuhopper`, 4 hours;
- multi-Re feature generation/per-Re statistics: `179412441.gadi-pbs`,
  `gpuvolta`, 2 hours;
- multi-Re residual diffusion: `179412442.gadi-pbs`, `gpuhopper`, 6 hours.

The replacement jobs retain the same strict `afterok` order. The initial
pending job IDs `179408457`--`179408461` were deleted before execution.

## Final outcome

All six replacement-chain stages completed successfully. The four trainers
reached epoch 9 and produced held-out smoke metrics:

- single-Re scOT selected epoch 9; test denormalized relative L2
  `0.04378938674926758`;
- single-Re residual PHASE selected epoch 5; test denormalized relative L2
  `0.14685158282518387`;
- multi-Re scOT selected epoch 9; test denormalized relative L2
  `0.05590289264917374`; and
- multi-Re residual PHASE selected epoch 5; test denormalized relative L2
  `0.044013211783021686`.

The diffusion stages used residual targets for all four fields. The multi-Re
diffusion log confirms a model-weights-only warm start from the single-Re
residual checkpoint, with epoch and optimizer state reset. These metrics are
acceptance diagnostics over reduced held-out subsets and are not paper results.

The companion six-ablation smoke suite completed tFNO, previous-DINO, scOT
without transfer learning, and scOT with transfer learning. The initial naive
and gated multi-Re jobs encountered the then-current 100-epoch production
guard; their retries were terminated before training. The current source has a
strict acceptance-only exception requiring exactly ten epochs and fractions
`[0.2, 0.1, 0.1]`, covered by unit and static-config tests. Rerunning those two
training smokes is optional release follow-up rather than evidence for any
reported scientific result.
