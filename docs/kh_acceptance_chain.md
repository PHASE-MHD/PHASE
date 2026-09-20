# Kelvin-Helmholtz end-to-end acceptance chain

This reduced experiment checks the complete canonical PHASE training path for
the periodic Kelvin-Helmholtz problem. It is an integration test, not a
replacement for the 100-epoch paper runs.

## Scope

The dependent chain runs:

1. single-Re `Re=Rm=1000` four-channel scOT;
2. single-Re scOT feature generation and train-only paired statistics;
3. single-Re four-channel residual diffusion from random initialization;
4. ten-regime multi-Re scOT, warm-started model-only from stage 1;
5. multi-Re feature generation and train-only per-Re paired statistics; and
6. multi-Re residual diffusion, warm-started model-only from stage 3.

Every training stage runs for 10 epochs. The canonical seed-42 split is formed
before deterministic reduction: 20% of training and 10% of each held-out
split are retained. This gives 160/10/10 trajectories for the single-Re run
and 1,600/100/100 trajectories for the ten-regime run.

## Locked KH behavior

The acceptance recipes retain `t=[0,5]`, `sub_t=5`, 51 frames, time-local KH
losses, paired magnetic normalization, balanced ten-regime scOT batches,
four-field residual targets, full-field Helmholtz projection, and diffusion
without Re conditioning. Only sample fractions, epochs, validation overhead,
artifact paths, and the explicit acceptance marker differ from production.

## Running on Gadi

```bash
cd /path/to/PHASE
bash scripts/submit_kh_acceptance_chain.sh
```

Set the machine-specific `DATA_ROOT`, `POSEIDON_ROOT`, `POSEIDON_DEPS`,
`MHD_PINO_DEPS`, `CONTAINER`, and PBS `PROJECT` variables explicitly.
`RUN_ROOT` and `QUEUE` may also be overridden. Set `DRY_RUN=1` to emit PBS
scripts without submitting jobs.

The launcher records job IDs in `outputs/kh_20pct_10ep/submitted_jobs.txt`.
Downstream stages use `afterok`, so a failure prevents dependent work from
starting and artifacts cannot be silently mixed with a rerun.

Default resource limits are 6 h, 2 h, 2 h, 8 h, 2 h, and 10 h for the six
stages respectively. Feature/statistics jobs use `gpuvolta`; training uses
`gpuhopper` unless `QUEUE` is overridden.
