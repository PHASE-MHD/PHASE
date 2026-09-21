# Scripts

# Previous baselines
- `train_tfno.py`: train the no warm-start tFNO baseline (based on previous work by Rosofsky and Huerta (2023)).
- `train_dino.py`: train the full-field EDM diffusion baseline (based on previous work by Kacmaz et al. (2025)).
- `generate_diffusion_features.py`: generate full-field DINO conditioner.

# scOT and PHASE scripts
- `prepare_data.py`: convert Dedalus HDF5 trajectories and vector-potential
  arrays to PHASE `.npy` layouts.
- `compute_statistics.py`: compute train-only trajectory or diffusion
  normalization statistics.
- `train_scot.py`: train the three-channel scOT ablations - no transfer learning ablation, ablation with POSEIDON transfer, naive multi-regime, and gated-
  adapter multi-regime ablations.
- `generate_scot_diffusion_features.py` exports physical-unit scOT/DNS pairs for
single- or multi-regime residual diffusion. 

# Evaluation scripts
- `evaluate_error.py`: run the unified held-out-test evaluator for tFNO, DINO,
  deterministic scOT, or PHASE. It requires an explicit Reynolds number and writes
  JSON, aggregate CSV, per-sample CSV, and text reports.
- `visualize.py`: generate held-out-test DNS/model/error fields, turbulence
  spectra and RMS-normalized PDFs, or KH passive-tracer post-processing. It
  requires an explicit Re and sample ID.