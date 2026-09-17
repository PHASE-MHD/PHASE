# PHASE copy manifest

Status: source inventory frozen; files have not yet been copied into the
unified package.

This is the literal legacy-source manifest for rebuilding the public PHASE
repository from MHD data through preprocessing, deterministic training,
diffusion-feature generation, diffusion training, and held-out-test
evaluation. It covers:

1. tFNO;
2. DINO;
3. scOT without POSEIDON transfer learning;
4. scOT with POSEIDON transfer learning;
5. naive multi-regime scOT;
6. gated-adapter multi-regime scOT;
7. four-channel multi-regime scOT with Helmholtz projection and physics loss;
8. the preceding model plus residual diffusion;
9. best single-Re DT scOT at Re=Rm=1000;
10. best single-Re DT scOT + diffusion at Re=Rm=1000;
11. selected single- and multi-Re KH scOT runs;
12. selected single- and multi-Re KH diffusion runs.

## Action vocabulary

- **COPY**: preserve the implementation, then update package imports/paths.
- **MERGE**: combine behavior from the named legacy files into one public file.
- **REFERENCE**: retain as provenance; do not ship as runtime source.
- **SYNTHESIZE**: write a clean path-independent replacement rather than
  copying a machine- or restart-specific file.
- **EXCLUDE**: not required by the requested experiments.

## Destination design

```text
PHASE/
  pyproject.toml
  src/phase/
    data/
    diffusion/
    losses/
    models/
    normalization/
    optim/
    physics/
    training/
    evaluation/
    visualization/
  scripts/
  configs/
    previous_baseline/{tfno,dino}/
    turbulence/{single_re,multi_re}/
    kh/{single_re,multi_re}/
    ablations/
  provenance/{ablations,runs}/
  tests/
```

The public tree uses one package. Do not preserve a top-level Python package
named `src`.

## 1. Canonical shared runtime from DINOs

These files form the runtime base for tFNO, DINO, and diffusion. All are
**COPY**, except files later marked **MERGE**.

### Entry point

- `DINOs/run_training.py`
- `DINOs/src/__init__.py`

### Activations

- `DINOs/src/activations/__init__.py`
- `DINOs/src/activations/activation_factory.py`
- `DINOs/src/activations/advanced.py`
- `DINOs/src/activations/composed.py`
- `DINOs/src/activations/parameterized.py`
- `DINOs/src/activations/standard.py`

### Data

- `DINOs/src/data/__init__.py` (**MERGE**)
- `DINOs/src/data/diffusion_dataset.py` (**MERGE**, canonical diffusion dataset)
- `DINOs/src/data/neurops_dataset.py` (**MERGE**)
- `DINOs/src/data/neurops_embedset.py`

### Diffusion

- `DINOs/src/diffusion/__init__.py`
- `DINOs/src/diffusion/models/__init__.py`
- `DINOs/src/diffusion/models/diffusion_factory.py`
- `DINOs/src/diffusion/models/elucidated_diffusion.py`
- `DINOs/src/diffusion/models/helmholtz_projection.py`
- `DINOs/src/diffusion/models/backbones/__init__.py`
- `DINOs/src/diffusion/models/backbones/unet.py`
- `DINOs/src/diffusion/models/components/__init__.py`
- `DINOs/src/diffusion/models/components/attention.py`
- `DINOs/src/diffusion/models/components/blocks.py`
- `DINOs/src/diffusion/models/components/layers.py`
- `DINOs/src/diffusion/models/components/normalization.py`
- `DINOs/src/diffusion/utils/__init__.py`
- `DINOs/src/diffusion/utils/utils.py`

### Losses

The current loss factory discovers modules dynamically, so retain this whole
set until registration is changed to explicit imports.

- `DINOs/src/losses/__init__.py` (**MERGE**)
- `DINOs/src/losses/loss_factory.py`
- `DINOs/src/losses/lp_loss.py`
- `DINOs/src/losses/physics_informed.py` (**MERGE**)
- `DINOs/src/losses/standard.py`
- `DINOs/src/losses/weighted.py`

### tFNO baseline

- `DINOs/src/neurops/__init__.py` (**MERGE**)
- `DINOs/src/neurops/model_factory.py` (**MERGE**)
- `DINOs/src/neurops/tfno.py`
- `DINOs/src/neurops/layers/__init__.py`
- `DINOs/src/neurops/layers/spectral_layers.py`

Do not copy `fno.py`, `tfno_neuralop.py`, or `uno_neuralop.py` into the
minimal release. Rewrite the model factory so it does not attempt to import
them.

### Normalization

- `DINOs/src/normalizations/__init__.py` (**MERGE**)
- `DINOs/src/normalizations/normalization_factory.py`
- `DINOs/src/normalizations/identity_norm.py`
- `DINOs/src/normalizations/min_max_norm.py` (**MERGE**, retain paired and per-Re modes)
- `DINOs/src/normalizations/physics_norm.py` (**MERGE**)
- `DINOs/src/normalizations/standard_norm.py`

### Optimizers and schedulers

- `DINOs/src/optimizers/__init__.py`
- `DINOs/src/optimizers/optimizer_factory.py`
- `DINOs/src/optimizers/scheduler_factory.py`
- `DINOs/src/optimizers/standard_optimizers.py`
- `DINOs/src/optimizers/standard_schedulers.py`

### Physics

- `DINOs/src/physics/__init__.py` (**MERGE**)
- `DINOs/src/physics/constraints.py`
- `DINOs/src/physics/pde_solvers.py` (**MERGE**)

### Training

- `DINOs/src/training/__init__.py` (**MERGE**)
- `DINOs/src/training/train_diffusion.py`
- `DINOs/src/training/train_neural_operator.py` (**MERGE**)
- `DINOs/src/training/trainers/__init__.py`
- `DINOs/src/training/trainers/base_neurops_trainer.py`
- `DINOs/src/training/trainers/diffusion_trainer.py`
- `DINOs/src/training/trainers/distributed_neurops_trainer.py`
- `DINOs/src/training/validators/__init__.py` (**MERGE**)
- `DINOs/src/training/validators/diffusion_validator.py`
- `DINOs/src/training/validators/divergence_validator.py`
- `DINOs/src/training/validators/neurops_validator.py` (**MERGE**)
- `DINOs/src/training/train_utils/__init__.py`
- `DINOs/src/training/train_utils/amp.py`
- `DINOs/src/training/train_utils/checkpoints.py`
- `DINOs/src/training/train_utils/early_stopping.py`
- `DINOs/src/training/train_utils/gradient_accumulation.py`
- `DINOs/src/training/train_utils/lr_finder.py`
- `DINOs/src/training/train_utils/state.py`

### Utilities

- `DINOs/src/utils/__init__.py` (**MERGE**)
- `DINOs/src/utils/batches.py` (**MERGE**)
- `DINOs/src/utils/config.py`
- `DINOs/src/utils/data_utils.py` (**MERGE**)
- `DINOs/src/utils/diffusion_tensor_normalization.py`
- `DINOs/src/utils/fourier_utils.py`

### Visualization

- `DINOs/src/visualization/__init__.py` (**MERGE**)
- `DINOs/src/visualization/diffusion_visualizer.py`
- `DINOs/src/visualization/enhanced_diffusion_visualizer.py`
- `DINOs/src/visualization/prediction_plots.py` (**MERGE**)
- `DINOs/src/visualization/spectra_plots.py` (**MERGE**)

## 2. Canonical scOT and multi-regime overlay

Use these files from `MHD-World-new` as the canonical scOT behavior. They
are **MERGE** into the shared package, replacing older DINOs behavior where
the same destination exists.

### Data and metadata

- `MHD-World-new/src/data/multi_re_neurops_dataset.py`
- `MHD-World-new/src/data/neurops_dataset.py`
- `MHD-World-new/src/data/__init__.py`
- `MHD-World-new/src/utils/batches.py`
- `MHD-World-new/src/utils/data_utils.py`
- `MHD-World-new/src/utils/__init__.py`

### scOT models

- `MHD-World-new/src/neurops/poseidon_mhd_finetune.py`
- `MHD-World-new/src/neurops/poseidon_mhd_re_finetune.py`
- `MHD-World-new/src/neurops/model_factory.py`
- `MHD-World-new/src/neurops/__init__.py`

Do not copy `poseidon_magnetic_scot_legacy_shared_head.py`. Keep
`poseidon_magnetic_scot.py` only if checkpoint-load testing shows that a
reported checkpoint contains keys from that class; none of the canonical YAMLs
select it directly.

### Direct-B physics and normalization

- `MHD-World-new/src/losses/__init__.py`
- `MHD-World-new/src/losses/physics_informed.py`
- `MHD-World-new/src/physics/__init__.py`
- `MHD-World-new/src/physics/pde_solvers.py`
- `MHD-World-new/src/normalizations/__init__.py`
- `MHD-World-new/src/normalizations/physics_norm.py`

### scOT training and diagnostics

- `MHD-World-new/src/training/train_neural_operator.py`
- `MHD-World-new/src/training/validators/__init__.py`
- `MHD-World-new/src/training/validators/neurops_validator.py`
- `MHD-World-new/src/visualization/prediction_plots.py`
- `MHD-World-new/src/visualization/spectra_plots.py`
- `MHD-World-new/src/visualization/__init__.py`

## 3. Required fork overlays

These are not separate public packages.

### No-transfer-learning overlay

Merge the scratch/random-initialization differences from:

- `MHD-World-scratch/run_training.py`
- `MHD-World-scratch/src/neurops/poseidon_mhd_finetune.py`
- `MHD-World-scratch/src/neurops/poseidon_mhd_re_finetune.py`
- `MHD-World-scratch/src/neurops/model_factory.py`

The public API must select pretrained versus random initialization explicitly
from YAML. It must not maintain a second scOT implementation.

### Naive multi-regime overlay

- `MHD-World-Re-naive/src/neurops/poseidon_mhd_re_input_finetune.py`
- `MHD-World-Re-naive/src/neurops/model_factory.py`
- `MHD-World-Re-naive/src/losses/physics_informed.py`
- `MHD-World-Re-naive/src/training/train_neural_operator.py`

Merge only the naive input-conditioning registration and any missing metadata
plumbing. Use the newer MHD-World-new implementations for shared code.

### KH-only single-Re overlay

Keep these changes opt-in:

- `MHD-World/scripts/calculate_statistics.py`
- `MHD-World/src/losses/physics_informed.py`
- `MHD-World/src/training/train_neural_operator.py`
- `MHD-World/src/visualization/prediction_plots.py`

They supply time-local/time-weighted component losses and corrected all-time
KH diagnostics. They must not alter default DT behavior.

## 4. Preprocessing and feature generation

### MHD trajectory conversion/statistics

- `MHD-World-new/scripts/convert_dedalus_h5_to_poseidon_bfield_npy.py`
- `MHD-World/scripts/convert_dedalus_h5_to_poseidon_npy.py`
- `MHD-World/scripts/convert_poseidon_vecpot_to_bfield_npy.py`
- `MHD-World-new/scripts/calculate_statistics.py`
- `MHD-World-new/scripts/KH_multiRe/compute_KH_multiRe_bfield_norms.py`
- `MHD-World-new/scripts/KH_multiRe/compute_KH_multiRe_bfield_norms_singleReCompatible.py`

Consolidate these into `scripts/prepare_data.py` and
`scripts/compute_statistics.py`; preserve split seed and train-only
statistics.

### Diffusion features

- `DINOs/data_generation/diffusion/generate_diffusion_features_pino.py`
- `DINOs/data_generation/diffusion/generate_diffusion_features_poseidon_vecpot.py`
- `DINOs/data_generation/diffusion/generate_diffusion_features_poseidon_multi_re_bfield.py`
- `DINOs/data_generation/diffusion/generate_diffusion_features_poseidon_kh_bfield.py`
- `DINOs/data_generation/diffusion/generate_unseen_re_scot_bfield_features.py`
- `DINOs/prev_SOTA/diffusion/Re1000/job_scripts/generate_official_tfno_diffusion_features.py`

Consolidate these into one public `scripts/generate_features.py`. Preserve
conditioner prediction, DNS target, Re/Rm, split, and source sample ID.
Require an explicit `prediction_mode: direct|residual`.

### Diffusion statistics

- `DINOs/scripts/calculate_statistics.py`

Retain separate conditioner and target statistics, residual-target statistics,
per-Re statistics, and train-only fitting.

## 5. Evaluation source

Move these into a unified, test-default evaluation package. They are source
despite their current legacy locations.

- `MHD-World-new/scripts/evaluate_poseidon_magnetic_scot_checkpoint.py`
- `MHD-World-new/scripts/evaluate_poseidon_mhd_re_finetune_checkpoint.py`
- `MHD-World-scratch/scripts/evaluate_poseidon_mhd_vecpot_derived_error.py`
- `DINOs/prev_SOTA/tFNO/dinos_repro/evaluate_tfno_vecpot_derived_error.py`
- `DINOs/evaluate_error.py`
- `DINOs/analysis_scripts/evaluate_error_multi_re_diffusion.py`
- `DINOs/analysis_scripts/evaluate_kh_diffusion_basic.py`
- `DINOs/scripts/validate_divergence.py`

The public evaluator must use one shared Fourier convention and report:

- per-field relative L2 and MSE for ux, uy, Bx, By, vorticity, and current;
- velocity and magnetic divergence MSE/RMS;
- low- and high-k spectrum errors;
- RMS-normalized PDF relative MAE;
- relative standard-deviation error;
- absolute kurtosis error;
- split, sample count, source sample IDs, config, checkpoint, and checkpoint
  epoch.

Default split is `test`. Validation evaluation must require an explicit flag.
KH evaluation may disable spectra/PDF metrics through config, but it must not
silently change derivative conventions.

Optional KH tracer/field diagnostics:

- `MHD-World/scripts/plot_poseidon_kh_tracer_timeslices.py`
- `MHD-World/scripts/make_poseidon_kh_movies.py`
- `DINOs/analysis_scripts/plot_kh_diffusion_fields_tracer.py`

## 6. Exact canonical legacy configs

These are **REFERENCE** inputs. Create path-independent **SYNTHESIZE** copies
at the destinations shown.

### Previous baselines

- `DINOs/configs/prev_SOTA/tFNO/dinos_repro/config_tfno_Re1000_prev_SOTA_dinos_repro_100ep.yaml`
  -> `configs/previous_baseline/tfno/re1000.yaml`
- `DINOs/prev_SOTA/diffusion/Re1000/configs/config_official_tfno_conditioner_Re1000_for_diffusion_features.yaml`
  -> `configs/previous_baseline/dino/conditioner_re1000.yaml`
- `DINOs/prev_SOTA/diffusion/Re1000/configs/config_diffusion_Re1000_prev_SOTA_full_from_official_tfno_epoch100_rerun.yaml`
  -> `configs/previous_baseline/dino/re1000.yaml`

### Three-channel scOT ablations

- `MHD-World-scratch/configs/config_poseidon_mhd_finetune_Re1000_scratch_physics_nueta1e3_corrected.yaml`
  -> `configs/ablations/scot_without_transfer.yaml`
- `MHD-World/configs/config_poseidon_mhd_finetune_Re1000_vecpot_TL_bs16_nueta1e3_corrected_100ep.yaml`
  -> `configs/ablations/scot_with_transfer.yaml`
- `MHD-World-Re-naive/configs/gadi/config_poseidon_mhd_re_input_vecpot_TL_corrected_Re1000_warm_bs1_100ep.yaml`
  -> `configs/ablations/naive_multi_regime.yaml`
- `MHD-World-new/configs/config_poseidon_mhd_re_vecpot_TL_corrected_Re1000_warm_FiLMAdapter_bs1_100ep.yaml`
  -> `configs/ablations/gated_adapter_multi_regime.yaml`

### Four-channel DT models

- `MHD-World/configs/config_poseidon_mhd_finetune_bfield_p99_bheavy_batch16_100ep_4h30_plots.yaml`
  -> `configs/turbulence/single_re/scot_re1000.yaml`
- `DINOs/configs/model_mag/poseidon_bfield/config_diffusion_Re1000_mag_helmholtz_poseidon_bfield_p99_bheavy_100ep.yaml`
  -> `configs/turbulence/single_re/diffusion_re1000.yaml`
- `MHD-World-new/configs/config_poseidon_mhd_re_finetune_bfield_p99_bheavy_warm_deep_adapters_gadi_continue48h.yaml`
  -> `configs/turbulence/multi_re/scot.yaml`
- `DINOs/configs/model_mag/poseidon_multi_re_bfield_full_physics_epoch91_residual_no_recond_resume_epoch45_per_re_norm_full_fullfieldproj_diagfix_pdf_48h_lr5e5/config_diffusion_multiRe_helmholtz_poseidon_bfield_full_physics_epoch91_residual_no_recond_resume_epoch45_per_re_norm_full_fullfieldproj_diagfix_pdf_48h_lr5e5.yaml`
  -> `configs/turbulence/multi_re/residual_diffusion.yaml`

The public multi-Re scOT and diffusion configs must describe fresh
epoch-0-to-100 runs. Do not encode continuation epochs in canonical YAML.

### KH models

- `MHD-World/configs/config_poseidon_mhd_finetune_KH_Re1000_bfield_p99_bheavy_alltimerel_batch1_subt5_100ep_resume_epoch58_48h.yaml`
  -> `configs/kh/single_re/scot_re1000.yaml`
- `DINOs/configs/model_mag/singleRe_KH_diffusion_best/config_diffusion_KH_Re1000_bestscot_residual_fullfield_100ep.yaml`
  -> `configs/kh/single_re/residual_diffusion_re1000.yaml`
- `MHD-World-new/configs/KH_multiRe/config_poseidon_mhd_re_finetune_KH_bfield_globalP99_bheavy_warm_deep_adapters_t0_4_100ep_48h_tinyval.template.yaml`
- `MHD-World-new/configs/KH_multiRe/config_poseidon_mhd_re_finetune_KH_bfield_globalP99_bheavy_warm_deep_adapters_t0_4_100ep_48h_tinyval_restart_epoch75_denormrel.yaml`
  -> merge into `configs/kh/multi_re/scot_t0_4.yaml`
- `DINOs/configs/model_mag/KH_multiRe_diffusion_globalP99_t0_4_epoch85_no_recond_warmstart_Re1000_per_re_norm_workers8_48h/config_diffusion_KH_multiRe_globalP99_t0_4_epoch85_residual_no_recond_warmstart_Re1000_per_re_norm_workers8_48h.yaml`
- `DINOs/configs/model_mag/KH_multiRe_diffusion_globalP99_t0_4_epoch85_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch80_48h/config_diffusion_KH_multiRe_globalP99_t0_4_epoch85_residual_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch80_48h.yaml`
  -> merge into `configs/kh/multi_re/residual_diffusion_t0_4.yaml`

## 7. Provenance records

Copy these verbatim under `provenance/ablations/`:

- `paper_draft_plots/ablation_table/tFNO.txt`
- `paper_draft_plots/ablation_table/DINO.txt`
- `paper_draft_plots/ablation_table/scOT_without_TL.txt`
- `paper_draft_plots/ablation_table/scOT_with_TL.txt`
- `paper_draft_plots/ablation_table/MR_NaiveRe.txt`
- `paper_draft_plots/ablation_table/MR_FiLMAdapter.txt`
- `paper_draft_plots/ablation_table/MR_4ch_HP_physics.txt`
- `paper_draft_plots/ablation_table/MR_4ch_HP_physics_plus_diffusion.txt`

Copy or generate run manifests under `provenance/runs/` for:

- DT single-Re scOT;
- DT single-Re scOT + diffusion;
- DT multi-Re scOT;
- DT multi-Re scOT + residual diffusion;
- KH single-Re scOT;
- KH single-Re scOT + residual diffusion;
- KH multi-Re scOT;
- KH multi-Re scOT + residual diffusion.

Each run manifest must record dataset identity/split seed, config, warm-start
source, checkpoint-selection metric, selected epoch, normalization statistics,
sampling steps, software versions, and legacy artifact locations.

## 8. Dependencies and packaging

Use these as dependency references, then synthesize one pinned
`pyproject.toml` and lock file:

- `DINOs/requirements_pip_gh200.txt`
- `MHD-World-new/requirements_poseidon_no_torch.txt`
- `MHD-World-new/requirements_pip_gh200.txt`

POSEIDON weight acquisition and licensing remain documented in
`README_poseidon.md`; do not copy the `poseidon/` development repository.

## 9. Files explicitly excluded

- all `logs/`, `checkpoints/`, and generated analysis outputs;
- PBS launchers;
- archive and smoke-test configs after canonical configs are synthesized;
- paper-only plotting scripts;
- `poseidon_magnetic_scot_legacy_shared_head.py`;
- spectral Re-conditioning experiments;
- alternative FNO/UNO implementations not selected by the eight recipes;
- duplicated MHD-World diffusion implementation;
- diffusion vorticity/current ablation losses as defaults;
- circular-padding ablations as defaults.

## 10. Required verification before legacy retirement

1. Import-test every public module in a clean environment.
2. Load each of the eight canonical checkpoints with strict key auditing.
3. Load the four selected DT/KH scOT/diffusion checkpoint families.
4. Confirm direct versus residual target mode for every diffusion recipe.
5. Confirm all validation/evaluation/plot sampling uses 32 diffusion steps.
6. Confirm train/validation/test source IDs and seeds match legacy splits.
7. Confirm all normalization statistics are fitted on training data only.
8. Numerically test paired normalization plus Helmholtz projection.
9. Numerically test velocity/B divergence after full-field reconstruction.
10. Compare a small held-out batch against each legacy implementation.
11. Run one CPU import/data smoke test and one GPU forward/backward smoke test
    for every distinct architecture/mode.
12. Reproduce the eight paper-table metric files from held-out test data.

## 11. Unresolved provenance gates

- The downloaded official `tfno_Re1000.pt` does not encode its original
  training job or epoch.
- The selected KH multi-Re diffusion directory names an epoch-85 conditioner,
  while the visible frozen source checkpoint is named epoch 75.
- The best historical single-Re DT diffusion YAML omits
  `prediction_mode`; current code defaults it to direct/full-field.
- `poseidon_magnetic_scot.py` is not selected by canonical configs, but
  checkpoint-key inspection is still required before excluding it absolutely.

No legacy source should be deleted until these gates and the verification
checklist are complete.
