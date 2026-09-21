# Configurations
SR / single_re = single-regime
MR / multi_re = multi-regime

Previous baseline configs:
- `previous_baseline/tfno/re1000.yaml`: three-channel Re=Rm=1000 tFNO trained from scratch without warm start
- `previous_baseline/dino/conditioner_re1000.yaml`: tFNO conditioner used to generate DINO features
- `previous_baseline/dino/re1000.yaml`: full-field DINO conditioned on the Re=Rm=1000 tFNO

Ablation configs:
- `ablations/scot_without_tl/re1000.yaml`: three-channel scOT trained from scratch at `Re=Rm=1000`
- `ablations/scot_with_tl/re1000.yaml`: three-channel scOT initialized with transfer learning from POSEIDON weights `camlab-ethz/Poseidon-T` at `Re=Rm=1000`
- `ablations/naive_multi_regime/multi_re.yaml`: warm-started three-channel
  scOT with naive Re/Rm input maps
- `ablations/gated_adapter_multi_regime/multi_re.yaml`: warm-started three-channel
  scOT with gated adapter Re/Rm conditioning
Turbulence configs:
- `turbulence/single_re/scot_re1000.yaml`: four-channel Re=Rm=1000 scOT with
  POSEIDON transfer learning, Helmholtz projection, and physics losses
- `turbulence/multi_re/scot.yaml`: four-channel MR scOT with Helmholtz
  projection and MHD, vorticity, and current losses
- `turbulence/single_re/phase_re1000.yaml`: corrected four-field SR PHASE residual diffusion
- `turbulence/multi_re/phase.yaml`: MR PHASE residual diffusion with per-Re
  paired normalization and its weights-only SR warm start.

Kelvin-Helmholtz instability configs:
- `kh/single_re/scot_re1000.yaml`: Re=Rm=1000 KH scOT
- `kh/multi_re/scot_t0_5.yaml`: multi-regime KH scOT with global paired P99 normalization, gated adapters, and full-validation checkpointing.
- `kh/single_re/phase_re1000.yaml`: four-field KH SR PHASE residual diffusion from random weights.
- `kh/multi_re/phase.yaml`: ten-regime KH MR PHASE residual diffusion with per-Re paired normalization and a weights-only single-Re warm start.
