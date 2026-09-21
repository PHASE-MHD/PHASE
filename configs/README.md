# Configurations
SR / single_re = single-regime
MR / multi_re = multi-regime

Ablation configs:
- `ablations/scot_without_tl/re1000.yaml`: three-channel scOT trained from scratch at `Re=Rm=1000`
- `ablations/scot_with_tl/re1000.yaml`: three-channel scOT initialized with transfer learning from POSEIDON weights `camlab-ethz/Poseidon-T` at `Re=Rm=1000`
- `ablations/naive_multi_regime/multi_re.yaml`: warm-started three-channel
  scOT with naive Re/Rm input maps

Turbulence configs:
- `turbulence/single_re/phase_re1000.yaml`: corrected four-field SR PHASE residual
  diffusion from random initialization.
- `turbulence/multi_re/phase.yaml`: MR PHASE residual diffusion with per-Re
  paired normalization and its weights-only SR warm start.

Kelvin-Helmholtz instability configs:
- `kh/single_re/scot_re1000.yaml`: Re=Rm=1000 KH SR PHASE
- `kh/multi_re/scot_t0_5.yaml`: multi-regime KH MR PHASE with global paired P99 normalization, gated adapters, and full-validation checkpointing.
- `kh/single_re/residual_diffusion_re1000.yaml`: four-field KH SR PHASE residual diffusion from random weights.
- `kh/multi_re/residual_diffusion_t0_5.yaml`: ten-regime KH MR PHASE residual diffusion with per-Re paired normalization and a weights-only single-Re warm start.
