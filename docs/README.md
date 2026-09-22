# Documentation

Use these guides to reproduce PHASE from prepared MHD trajectories.

1. Read [Data format](data_format.md) and [Data preparation](preprocessing.md).
2. Reproduce the previous [tFNO](previous_baseline_tfno.md) and
   [DINO](previous_baseline_dino.md) baselines.
3. Run the scOT ablations:
   [without transfer learning](scot_without_transfer_learning.md),
   [with transfer learning](scot_with_transfer_learning.md),
   [naive multi-regime conditioning](naive_multi_regime.md), and
   [gated-adapter conditioning](gated_adapter_multi_regime.md).
4. Train decaying-turbulence PHASE:
   [single-Re scOT](turbulence_single_re_scot.md),
   [multi-Re scOT](four_channel_hp_physics.md), and
   [residual diffusion](turbulence_residual_diffusion.md).
5. Train Kelvin--Helmholtz PHASE:
   [scOT](kh_scot.md) and [residual diffusion](kh_residual_diffusion.md).
6. Run held-out [evaluation](evaluation.md) and
   [visualization](visualization.md).

Canonical YAML files under `configs/` are the source of truth for model,
data, normalization, loss, optimization, and checkpoint settings. Training
data and released checkpoints are external artifacts.
