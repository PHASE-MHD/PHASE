# Configurations

Canonical, path-independent YAML configurations will be organized by previous
baseline, ablation, physical problem, and single- versus multi-regime scope.

Machine-specific data, output, and checkpoint roots must not be committed.

Implemented ablation configs:

- `ablations/scot_without_tl/re1000.yaml`: three-channel scOT trained from
  scratch at `Re=Rm=1000`, with the reported batch size of 1.
- `ablations/scot_with_tl/re1000.yaml`: three-channel scOT initialized from
  `camlab-ethz/Poseidon-T` at `Re=Rm=1000`, with batch size 16.
- `ablations/naive_multi_regime/multi_re.yaml`: warm-started three-channel
  scOT with naive constant Re/Rm input maps and balanced ten-regime batches.
- `turbulence/single_re/phase_re1000.yaml`: corrected four-field SR residual
  diffusion from random initialization.
- `turbulence/multi_re/phase.yaml`: reported MR residual diffusion with per-Re
  paired normalization and its historical weights-only SR warm start.
- `kh/single_re/scot_re1000.yaml`: canonical Re=Rm=1000 KH scOT on 51 frames.
- `kh/multi_re/scot_t0_5.yaml`: canonical ten-regime KH scOT with global paired P99 normalization, gated adapters, tiny monitoring, and full-validation checkpointing.
- `kh/single_re/residual_diffusion_re1000.yaml`: four-field KH residual diffusion from random weights.
- `kh/multi_re/residual_diffusion_t0_5.yaml`: ten-regime KH residual diffusion with per-Re paired normalization and a weights-only single-Re warm start.
- `acceptance/dt_20pct_10ep/`: reduced end-to-end DT acceptance recipes.
  These retain the canonical model recipes while using 20% of training data,
  10% of each held-out split, and 10 epochs. They are not paper-result
  configurations.
- `acceptance/kh_20pct_10ep/`: reduced canonical KH `t=[0,5]` recipes with 51 frames, 20% of training data, 10% of each held-out split, and 10 epochs.
