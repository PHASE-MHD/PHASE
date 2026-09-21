# PHASE file and source provenance

This README inventories the public repository directory by directory and file
by file. It records purpose, legacy origin, and intentional PHASE changes.
Generated, Git-ignored directories (`build/`, `outputs/`, `.pytest_cache/`,
`*.egg-info/`, and `__pycache__/`) are not release source and are excluded.

## Labels

- **Verbatim**: byte-identical legacy source, apart from a requested filename.
- **Adapted**: one legacy source with public imports/API or a documented fix.
- **Merged**: behavior consolidated from multiple legacy trees.
- **New**: written for this public repository.
- **Metadata**: configuration, documentation, provenance, or licensing.

`COPY_MANIFEST.md` and `provenance/batch_*.md` contain the detailed source
paths, hashes, decisions, and validation evidence. “Working tree” is stated
where canonical behavior included uncommitted legacy research changes.

## Root and CI

| File | Status | Purpose |
| --- | --- | --- |
| `.gitignore` | New | Excludes data, checkpoints, outputs, caches, and builds. |
| `.github/workflows/ci.yml` | New | Public CPU continuous-integration checks. |
| `README.md` | New | Project overview and documentation entry point. |
| `README_poseidon.md` | New | External POSEIDON weights and transfer-learning setup. |
| `README_src_changes.md` | New metadata | High-level legacy-source change record. |
| `README_file_provenance.md` | New metadata | This file-level inventory. |
| `COPY_MANIFEST.md` | New metadata | Authoritative legacy-to-public source map. |
| `THIRD_PARTY_NOTICES.md` | New metadata | Limited third-party attribution. |
| `LICENSE` | New metadata | MIT license. |
| `pyproject.toml` | New | Package metadata, dependencies, entry points, and pytest settings. |

## MHD_DataGeneration

The original legacy files remain unchanged.

| File | Status | Purpose |
| --- | --- | --- |
| `dedalus_mhd_turbulence_parallel.py` | Verbatim from `PINOs/Dedalus/.../dedalus_mhd_parallel.py`; renamed | Periodic 2-D incompressible decaying-MHD trajectories. |
| `dedalus_mhd_kh_parallel.py` | Verbatim from the legacy Dedalus directory | Periodic double-shear Kelvin–Helmholtz MHD trajectories. |
| `my_random_fields.py` | Verbatim from the legacy Dedalus directory | Periodic Matérn/RBF Gaussian random initial fields. |
| `README.md` | New | Physics, dependencies, commands, parallelism, and HDF5 outputs. |

No numerical source behavior was changed in these three generators.

## configs

All YAMLs are **New path-independent reconstructions** of audited legacy
configs. Absolute cluster paths and job/restart names were removed. Scientific
settings are retained unless the corresponding provenance batch records an
explicit correction.

| File | Purpose |
| --- | --- |
| `README.md` | Configuration index. |
| `previous_baseline/tfno/re1000.yaml` | No-warm-start Re=1000 tFNO baseline. |
| `previous_baseline/dino/conditioner_re1000.yaml` | Released tFNO conditioner contract. |
| `previous_baseline/dino/re1000.yaml` | Previous-study full-field DINO. |
| `ablations/scot_without_tl/re1000.yaml` | Three-channel random scOT, batch 1. |
| `ablations/scot_with_tl/re1000.yaml` | Three-channel POSEIDON-transfer scOT, batch 16. |
| `ablations/naive_multi_regime/multi_re.yaml` | Constant-map Re/Rm conditioning. |
| `ablations/gated_adapter_multi_regime/multi_re.yaml` | Gated FiLM/deep-adapter conditioning. |
| `turbulence/multi_re/scot.yaml` | Direct-B, projection, MHD, vorticity, and current ablation. |
| `turbulence/single_re/scot_re1000.yaml` | Canonical four-channel Re=1000 DT scOT. |
| `turbulence/single_re/phase_re1000.yaml` | Corrected residual SR PHASE. |
| `turbulence/multi_re/phase.yaml` | Reported residual MR PHASE. |
| `kh/single_re/scot_re1000.yaml` | Canonical Re=1000 KH scOT. |
| `kh/single_re/residual_diffusion_re1000.yaml` | Re=1000 KH residual diffusion. |
| `kh/multi_re/scot_t0_5.yaml` | Ten-regime global-P99 KH scOT, `t=[0,5]`. |
| `kh/multi_re/residual_diffusion_t0_5.yaml` | Ten-regime per-Re-normalized KH residual diffusion. |

All `.gitkeep` files are empty placeholders with no runtime behavior.

## src/phase

| File | Status | Purpose |
| --- | --- | --- |
| `__init__.py` | New | Public package/version declaration. |
| `config_validation.py` | New | Guards model modes, normalization, splits, KH time, warm starts, and paths. |

### activations

All are **Adapted from DINOs**; imports changed to `phase.*`, while activation
mathematics was retained.

| File | Purpose |
| --- | --- |
| `__init__.py` | Exports/registers activations. |
| `activation_factory.py` | Config-driven activation registry. |
| `standard.py` | Standard PyTorch activations. |
| `advanced.py` | Additional activations. |
| `parameterized.py` | Learnable activations. |
| `composed.py` | Activation compositions. |

### cli

| File | Status | Purpose |
| --- | --- | --- |
| `__init__.py` | New | CLI package marker. |
| `validate_configs.py` | New | Installed config-validation command. |

### data

| File | Status | Purpose and changes |
| --- | --- | --- |
| `__init__.py` | Merged | One explicit public dataset API. |
| `neurops_dataset.py` | Merged from MHD-World forks | Single-Re data, deterministic splits, residual inputs, metadata, and acceptance fractions. |
| `multi_re_neurops_dataset.py` | Adapted from MHD-World-new | Multi-Re data, Re/Rm metadata, balanced sampling, and fractions. |
| `neurops_embedset.py` | Adapted from DINOs | Embedded/operator baseline dataset compatibility. |
| `diffusion_dataset.py` | Merged from DINOs working states | Eager/lazy features, residual targets, IDs, per-Re normalization; fixes eager unseen-Re `re_per_sample`. |

### diffusion

| File | Status | Purpose |
| --- | --- | --- |
| `__init__.py` | Adapted from DINOs | Diffusion exports. |
| `models/__init__.py` | Adapted | Supported model exports. |
| `models/diffusion_factory.py` | Adapted from DINOs | Diffusion registry. |
| `models/elucidated_diffusion.py` | Merged from DINOs states | Explicit full-field DINO and residual PHASE EDM modes, metadata, optional derivative losses, and projection. |
| `models/helmholtz_projection.py` | Adapted/extended from DINOs | Fourier projection for velocity/B pairs and reconstructed residual fields. |
| `models/backbones/__init__.py` | Adapted from DINOs | Backbone exports. |
| `models/backbones/unet.py` | Adapted/extended from DINOs | Conditional EDM U-Net with PHASE conditioning/padding paths. |
| `models/components/__init__.py` | Adapted from DINOs | Component exports. |
| `models/components/attention.py` | Adapted from DINOs | U-Net attention. |
| `models/components/blocks.py` | Adapted from DINOs | Residual/resolution blocks. |
| `models/components/layers.py` | Adapted from DINOs | Convolutional and embedding layers. |
| `models/components/normalization.py` | Adapted from DINOs | Diffusion normalization layers. |
| `utils/__init__.py` | Adapted from DINOs | Diffusion utility exports. |
| `utils/utils.py` | Adapted from DINOs | Tensor/model helpers. |

### evaluation

All are **New consolidation code** informed by legacy `evaluate_error`
scripts. Public reports require the held-out test split.

| File | Purpose |
| --- | --- |
| `__init__.py` | Evaluation exports. |
| `inference.py` | tFNO/scOT inference, DINO sampling, and residual reconstruction. |
| `metrics.py` | Relative L2, MSE, spectra, PDFs, std, kurtosis, and aggregation. |
| `physics.py` | Common spectral derivatives, vorticity, current, and divergence. |
| `reporting.py` | JSON, aggregate/per-sample CSV, text reports, and artifact hashes. |

### losses

| File | Status | Purpose and changes |
| --- | --- | --- |
| `__init__.py` | Merged | Public loss registrations. |
| `loss_factory.py` | Adapted from DINOs | Config-driven loss registry. |
| `lp_loss.py` | Adapted from DINOs | Absolute/relative Lp losses. |
| `standard.py` | Adapted from DINOs | Standard loss wrappers. |
| `weighted.py` | Adapted from DINOs | Channel-weighted losses. |
| `physics_informed.py` | Merged from DINOs and MHD-World forks | Vector-potential/direct-B data, IC and PDE losses; vorticity/current and opt-in KH time-local reductions. |

### models

| File | Status | Purpose and changes |
| --- | --- | --- |
| `__init__.py` | Merged | Supported-model exports. |
| `model_factory.py` | Merged | Explicit tFNO/scOT construction; excluded experiments removed. |
| `tfno.py` | Adapted from DINOs | Previous-study tensorized FNO. |
| `checkpoint_mapping.py` | New | Maps released baseline checkpoint keys. |
| `scot_mhd.py` | Merged from MHD-World, new, and scratch | Random/pretrained initialization, 3/4 channels, magnetic initialization, residual rollout, parameter groups, projection. |
| `scot_mhd_naive_re.py` | Adapted from MHD-World-Re-naive | Standardized log10(Re/Rm) constant maps. |
| `scot_mhd_gated_re.py` | Adapted/merged from MHD-World-new | FiLM/deep gated adapters and final projection. |
| `layers/__init__.py` | Adapted from DINOs | Spectral-layer exports. |
| `layers/spectral_layers.py` | Adapted from DINOs | tFNO Fourier layers. |

### normalizations

| File | Status | Purpose and changes |
| --- | --- | --- |
| `__init__.py` | Merged | Normalizer registrations. |
| `normalization_factory.py` | Adapted from DINOs | Normalizer registry. |
| `identity_norm.py` | Adapted from DINOs | No-op normalization. |
| `standard_norm.py` | Adapted from DINOs | Mean/std normalization. |
| `min_max_norm.py` | Merged from DINOs states | Standard, paired, per-Re paired min-max, and unseen-Re log interpolation. |
| `physics_norm.py` | Merged from MHD-World-new/DINOs | POSEIDON velocity and paired component scaling compatible with projection. |

### optimizers

All are **Adapted from DINOs**. Imports changed; missing `torch` for the Lion
fallback and disabled-scheduler registration order were fixed.

| File | Purpose |
| --- | --- |
| `__init__.py` | Optimizer/scheduler exports. |
| `optimizer_factory.py` | Optimizer registry. |
| `scheduler_factory.py` | Scheduler registry. |
| `standard_optimizers.py` | Adam/AdamW/SGD and optional wrappers. |
| `standard_schedulers.py` | Multistep/cosine/plateau/disabled schedulers. |

### physics

| File | Status | Purpose and changes |
| --- | --- | --- |
| `__init__.py` | Merged | Physical-operator exports. |
| `constraints.py` | Adapted from DINOs | Velocity and magnetic divergence constraints. |
| `pde_solvers.py` | Merged from DINOs/MHD-World-new | Spectral MHD residuals for A and direct B with per-sample `nu`,`eta`. |

### preprocessing

These are **New consolidated implementations** of the legacy converters and
statistics scripts listed in `batch_03_preprocessing.md`.

| File | Purpose |
| --- | --- |
| `__init__.py` | Conversion/statistics exports. |
| `conversion.py` | Dedalus-to-array and spectral A-to-B conversion. |
| `statistics.py` | Chunked train-only, paired-P99, conditioner, and residual statistics. |

### training

These are **New consolidated loops** reproducing legacy behavior with guarded
public recipes.

| File | Purpose and changes |
| --- | --- |
| `__init__.py` | Lazy trainer exports. |
| `tfno_trainer.py` | Locked previous tFNO training/validation. |
| `scot_trainer.py` | scOT training, weight-only warm starts, parameter groups, full/tiny validation, best checkpoints. |
| `dino_trainer.py` | Full-field/residual diffusion, separate normalizers, physical validation, weight-only warm start versus resume. |

### utils

| File | Status | Purpose and changes |
| --- | --- | --- |
| `__init__.py` | Merged | Utility exports. |
| `config.py` | Adapted from DINOs | YAML/nested config helpers. |
| `batches.py` | Merged | Tensor plus Re/Rm/source-ID batch handling. |
| `data_utils.py` | Merged | Channel and normalization helpers. |
| `diffusion_tensor_normalization.py` | Merged from DINOs states | Separate conditioner/target transforms and residual reconstruction. |
| `fourier_utils.py` | Adapted/merged | Shared differentiable periodic Fourier operators. |

### visualization

| File | Status | Purpose |
| --- | --- | --- |
| `__init__.py` | New | Visualization exports. |
| `plots.py` | New consolidation of audited plotters | Correctly oriented held-out fields, normalized spectra/PDFs, and velocity-derived KH tracer plots with provenance. |

## scripts

All executable files are **New public wrappers** around `src/phase`; they
replace machine-specific legacy scripts and PBS launchers.

| File | Purpose |
| --- | --- |
| `README.md` | Command index. |
| `prepare_data.py` | HDF5/A/direct-B conversion CLI. |
| `compute_statistics.py` | Train-only statistics CLI. |
| `train_tfno.py` | tFNO training. |
| `train_scot.py` | Deterministic scOT training. |
| `train_dino.py` | Full-field DINO/residual PHASE training. |
| `generate_diffusion_features.py` | Full-field DINO conditioner/DNS features. |
| `generate_scot_diffusion_features.py` | Physical scOT/DNS pairs, IDs, and Re metadata. |
| `evaluate_error.py` | Unified test evaluation. |
| `visualize.py` | Unified test visualization. |
| `validate_configs.py` | Config audit wrapper. |
| `verify_artifact.py` | One-artifact digest verification. |
| `verify_artifact_manifest.py` | Canonical artifact verification and test exports. |
| `submit_ablation_smoke_suite.sh` | Reduced first-six-ablation jobs. |
| `submit_dt_acceptance_chain.sh` | Dependent DT acceptance chain. |
| `submit_kh_acceptance_chain.sh` | Dependent KH acceptance chain. |

## tests

Tests are **New PHASE release code** unless noted; a few factory primitives
were adapted from DINOs tests and rewritten for `phase.*`.

| File | Purpose |
| --- | --- |
| `README.md` | Test tiers, markers, and external checkpoint variables. |
| `conftest.py` | Shared fixtures and marker/runtime setup. |
| `fixtures/.gitkeep` | Empty synthetic-fixture placeholder. |

### tests/unit

| File | Purpose |
| --- | --- |
| `test_artifact_manifest.py` | Manifest schema, paths, sizes, hashes. |
| `test_config_validation.py` | All public YAML and recipe guards. |
| `test_data_runtime.py` | Splits, metadata, fractions, balanced batches. |
| `test_dino_baseline.py` | Full-field DINO contract. |
| `test_evaluation.py` | Derivatives, metrics, IDs, test-only reports. |
| `test_foundation_runtime.py` | Shared factories; partly adapted from DINOs tests. |
| `test_four_channel_scot.py` | Direct-B scOT, projection, physics, optimizer groups. |
| `test_gated_adapter_multi_regime.py` | Gated conditioning and warm-start identity. |
| `test_kh_acceptance.py` | Reduced KH chain guards. |
| `test_kh_residual_diffusion.py` | KH residual normalization/reconstruction/resume. |
| `test_kh_scot.py` | KH frames, times, losses, diagnostics. |
| `test_losses.py` | Standard/weighted/relative/physics losses. |
| `test_naive_multi_regime.py` | Naive Re/Rm conditioning. |
| `test_normalizations.py` | All normalization and unseen-Re interpolation. |
| `test_optimizers.py` | Optimizers and parameter groups. |
| `test_phase_residual_diffusion.py` | DT residual mode and full-field projection. |
| `test_preprocessing.py` | HDF5 mapping, A-to-B, splits, statistics. |
| `test_schedulers.py` | Scheduler modes. |
| `test_scot_with_tl.py` | POSEIDON transfer recipe. |
| `test_scot_without_tl.py` | Random-initialization recipe. |
| `test_tfno_baseline.py` | tFNO architecture/loss/optimizer. |
| `test_visualization.py` | Orientation, selection, plots, provenance. |

### tests/integration

| File | Purpose |
| --- | --- |
| `test_evaluation_inference.py` | Synthetic inference for all four model families. |
| `test_gpu_acceptance.py` | CUDA forward/backward and residual reconstruction. |
| `test_visualization_cli.py` | Synthetic visualization CLI contract. |

### tests/checkpoint_compatibility

All are **New gates** for external weights; checkpoints are not redistributed.

| File | Family |
| --- | --- |
| `test_tfno_legacy_checkpoint.py` | Previous tFNO. |
| `test_dino_legacy_checkpoints.py` | Released tFNO conditioner/full-field DINO. |
| `test_scot_without_tl_legacy_checkpoint.py` | No-TL scOT. |
| `test_scot_with_tl_legacy_checkpoint.py` | TL scOT. |
| `test_naive_multi_regime_legacy_checkpoint.py` | Naive-MR scOT. |
| `test_gated_adapter_multi_regime_legacy_checkpoint.py` | Gated-MR scOT. |
| `test_four_channel_scot_legacy_checkpoints.py` | Direct-B SR/MR scOT. |
| `test_residual_diffusion_legacy_checkpoints.py` | DT SR/MR diffusion. |
| `test_kh_scot_legacy_checkpoints.py` | KH SR/MR scOT. |
| `test_kh_residual_diffusion_legacy_checkpoints.py` | KH SR/MR diffusion. |

All test `.gitkeep` files are empty placeholders.

## docs

All are **New documentation** based on audited source/config/checkpoint/run
records.

| File | Purpose |
| --- | --- |
| `README.md` | Documentation index. |
| `data_format.md` | Arrays, channels, metadata, splits. |
| `preprocessing.md` | Conversion/statistics commands. |
| `previous_baseline_tfno.md` | tFNO reproduction. |
| `previous_baseline_dino.md` | DINO reproduction. |
| `scot_without_transfer_learning.md` | No-TL ablation. |
| `scot_with_transfer_learning.md` | TL ablation. |
| `naive_multi_regime.md` | Naive-MR ablation. |
| `gated_adapter_multi_regime.md` | Gated-MR ablation. |
| `turbulence_single_re_scot.md` | Best Re=1000 DT scOT. |
| `four_channel_hp_physics.md` | Direct-B/HP/physics ablation. |
| `turbulence_residual_diffusion.md` | DT SR/MR residual diffusion. |
| `kh_scot.md` | KH SR/MR deterministic models. |
| `kh_residual_diffusion.md` | KH SR/MR residual models. |
| `evaluation.md` | Metrics and evaluation. |
| `visualization.md` | Fields, spectra, PDFs, tracer. |
| `reproduction.md` | Clean-checkout execution order. |
| `release_checklist.md` | Release checks. |
| `dt_acceptance_chain.md` | Reduced DT chain. |
| `kh_acceptance_chain.md` | Reduced KH chain. |

## provenance

All are **New metadata**, not runtime source or weights.

| File | Purpose |
| --- | --- |
| `README.md` | Provenance index. |
| `checkpoint_manifest.yaml` | Canonical external paths, sizes, and SHA-256. |
| `batch_01_shared_runtime.md` | Shared runtime admission/fixes. |
| `batch_02_data_normalization.md` | Dataset/normalization hashes and merge. |
| `batch_03_preprocessing.md` | Conversion/statistics consolidation. |
| `batch_04_tfno_baseline.md` | tFNO reconstruction. |
| `batch_05_dino_baseline.md` | DINO reconstruction. |
| `batch_06_scot_without_tl.md` | No-TL reconstruction. |
| `batch_07_scot_with_tl.md` | TL reconstruction. |
| `batch_08_naive_multi_regime.md` | Naive-MR reconstruction. |
| `batch_09_gated_adapter_multi_regime.md` | Gated-MR reconstruction/audit. |
| `batch_10_four_channel_scot.md` | Direct-B/HP/physics admission. |
| `batch_11_residual_diffusion.md` | DT residual diffusion audit. |
| `batch_12_kh_scot.md` | KH scOT audit. |
| `batch_13_kh_residual_diffusion.md` | KH diffusion audit. |
| `batch_14_evaluation.md` | Evaluation implementation. |
| `batch_14b_visualization.md` | Visualization implementation. |
| `batch_15_release_hardening.md` | Packaging/config/CPU checks. |
| `batch_16_artifact_gpu_acceptance.md` | Artifacts/checkpoints/GPU checks. |
| `batch_17_dt_acceptance.md` | Reduced DT chain and outcomes. |
| `batch_18_kh_acceptance.md` | Reduced KH chain and outcomes. |

### provenance/ablations

Each **New metadata** file records the legacy config/checkpoint, split,
training length, reported row, and caveats: `tfno.md`, `dino.md`,
`scot_without_tl.md`, `scot_with_tl.md`, `naive_multi_regime.md`,
`gated_adapter_multi_regime.md`, and `four_channel_hp_physics.md`.

### provenance/runs

Each **New metadata** file records one canonical chain:
`turbulence_single_re_scot_re1000.md`,
`turbulence_single_re_residual_diffusion_re1000.md`,
`kh_single_re_scot_re1000.md`,
`kh_single_re_residual_diffusion_re1000.md`,
`kh_multi_re_scot_t0_5.md`, and
`kh_multi_re_residual_diffusion_t0_5.md`.

Provenance `.gitkeep` files are empty placeholders.

## environments

| File | Status | Purpose |
| --- | --- | --- |
| `README.md` | New | Installation and external POSEIDON notes. |
| `phase-cuda121.yml` | New | CUDA 12.1 model-training/evaluation environment; Dedalus is separate. |

## What PHASE changed in source code

PHASE is not only a directory repackaging:

1. Legacy `src.*` imports and machine paths became an installable `phase.*`
   package with explicit configuration.
2. Four MHD-World forks became one config-selected scOT implementation.
3. Direct `(u_x,u_y,B_x,B_y)` prediction, paired normalization, velocity/B
   Helmholtz projection, and direct-B PDE residuals were integrated.
4. Vorticity/current losses and opt-in KH time-local losses were integrated
   without changing DT defaults.
5. Full-field DINO and residual PHASE are explicit guarded modes; residual
   projection acts on the reconstructed full physical field.
6. Per-Re normalization and unseen-Re log interpolation were retained, and the
   eager unseen-Re metadata omission was fixed.
7. Weight-only warm starts were separated from true optimizer/epoch resumes.
8. Evaluation now enforces held-out test splits, stable IDs, common spectral
   derivatives, and provenance-rich reports.
9. Preprocessing, validation, artifact checks, and acceptance chains were made
   path-independent.
10. Dedalus generator numerics were not changed.

