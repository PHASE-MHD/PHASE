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
