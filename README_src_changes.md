# Source changes recovered from the legacy repositories

This manifest records the source and canonical configurations needed to
consolidate the eight paper ablations and the selected decaying-turbulence
(DT) and Kelvin-Helmholtz (KH) models into one public PHASE repository.
Job scripts, logs, checkpoints, analysis outputs, and paper-only plotting
scripts are excluded from the source inventory.

The exact file-by-file consolidation inventory is maintained in
`COPY_MANIFEST.md`. That manifest is authoritative for copying; this file
explains why the recovered changes are needed.

It was assembled from the working trees, saved git status reports, Git
history, source diffs, and the provenance records under
`paper_draft_plots/ablation_table/`. Git cannot identify the author of every
uncommitted line, so changed means present in the recovered working state.

The MHD-World directories are overlapping forks:

- `MHD-World`: canonical single-regime training and KH time-local losses.
- `MHD-World-new`: multi-regime gated adapters and four-channel PHASE.
- `MHD-World-scratch`: random scOT initialization for no transfer learning.
- `MHD-World-Re-naive`: naive Reynolds-number input conditioning.

They must be merged into one implementation, not copied wholesale.

## Ablation-to-source map

| Record | Legacy source | Required implementation |
| --- | --- | --- |
| `tFNO.txt` | `DINOs` | tFNO, vector-potential physics loss, training and validation |
| `DINO.txt` | `DINOs` | tFNO conditioning and full-field EDM diffusion |
| `scOT_without_TL.txt` | `MHD-World-scratch` | random scOT initialization, three-channel rollout |
| `scOT_with_TL.txt` | `MHD-World` | POSEIDON/scOT three-channel fine-tuning |
| `MR_NaiveRe.txt` | `MHD-World-Re-naive` | standardized log10(Re)/log10(Rm) input maps |
| `MR_FiLMAdapter.txt` | `MHD-World-new` | FiLM conditioning and gated residual adapters |
| `MR_4ch_HP_physics.txt` | `MHD-World-new` | direct four-field prediction, projection, MHD and derived losses |
| `MR_4ch_HP_physics_plus_diffusion.txt` | `DINOs` | residual diffusion, per-Re paired normalization, full-field projection |

## Decaying-turbulence run coverage

The public repository must also expose the two best single-regime DT recipes
at `Re=Rm=1000`. They are standalone reported models and provide the
initialization chain for multi-regime PHASE. The four-channel single-Re scOT
is not the three-channel `scOT_with_TL.txt` ablation.

| Model | Canonical legacy configuration | Required behavior |
| --- | --- | --- |
| Best single-Re scOT | `MHD-World/configs/config_poseidon_mhd_finetune_bfield_p99_bheavy_batch16_100ep_4h30_plots.yaml` | Four channels; POSEIDON velocity transfer; mean-velocity magnetic initialization; magnetic residual rollout; POSEIDON fluid normalization and paired magnetic scale; velocity/B Helmholtz projection; direct-B PDE, vorticity, and current losses; `nu=eta=1e-3`; batch 16; 100 epochs. Checkpoint: `MHD-World/checkpoints/poseidon_mhd_finetune_Re1000_bfield_p99_bheavy_batch16_100ep_4h30_plots.pt`. |
| Best single-Re scOT + diffusion | `DINOs/configs/model_mag/poseidon_bfield/config_diffusion_Re1000_mag_helmholtz_poseidon_bfield_p99_bheavy_100ep.yaml` | Four-channel EDM conditioned on scOT; separate paired min-max conditioner/target statistics; velocity/B projection; 32 sampling steps; no diffusion vorticity/current losses; batch 64; 100 epochs. Checkpoint: `DINOs/checkpoints/mag/poseidon_bfield_p99_bheavy/Diffusion_MHDmag_Re1000_helmholtz_poseidon_bfield_p99_bheavy_100ep.pt`. |

The single-Re diffusion YAML has no `prediction_mode` or `residual_target`.
The recovered dataset defaults such configs to direct/full-field prediction.
Document this historical run as full-field unless checkpoint-compatible
forensics prove otherwise. Its weights may warm-start multi-Re residual
diffusion, but that must be weight-only and reset epoch, optimizer, scheduler,
and checkpoint-selection state.

## Kelvin-Helmholtz run coverage

| Model | Canonical legacy configuration | Required behavior |
| --- | --- | --- |
| Single-Re scOT | `MHD-World/configs/config_poseidon_mhd_finetune_KH_Re1000_bfield_p99_bheavy_alltimerel_batch1_subt5_100ep_resume_epoch58_48h.yaml` | Four channels, paired P99 scale, projection, direct-B and derived losses, time-local relative losses; 100-epoch chain, best reported near epoch 95. |
| Single-Re scOT + diffusion | `DINOs/configs/model_mag/singleRe_KH_diffusion_best/config_diffusion_KH_Re1000_bestscot_residual_fullfield_100ep.yaml` | Residual learning for all four fields, separate conditioner/residual statistics, full-field projection, 32 steps. |
| Multi-Re scOT, global P99, t=[0,4] | `MHD-World-new/configs/KH_multiRe/config_poseidon_mhd_re_finetune_KH_bfield_globalP99_bheavy_warm_deep_adapters_t0_4_100ep_48h_tinyval_restart_epoch75_denormrel.yaml` | Ten regimes, Re=1000 warm start, gated adapters, global paired magnetic P99, time-local losses, full and tiny validation. |
| Multi-Re scOT + diffusion, t=[0,4] | `DINOs/configs/model_mag/KH_multiRe_diffusion_globalP99_t0_4_epoch85_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch80_48h/config_diffusion_KH_multiRe_globalP99_t0_4_epoch85_residual_no_recond_warmstart_Re1000_per_re_norm_workers8_restart_epoch80_48h.yaml` | Residual diffusion, no diffusion Re conditioning, per-Re paired conditioner/residual normalization, single-Re weight warm start, balanced regimes, full-field projection, 32 steps. |

Keep global/per-Re P99 and t=[0,5]/t=[0,4] as explicit KH ablations. The
paper-facing chain is global-P99 scOT on t=[0,4], then per-Re residual
diffusion. Reconcile the epoch-85 directory name with the visible epoch-75
frozen source checkpoint before publication.

## Required MHD-World source

### Shared single-Re path

- `src/neurops/poseidon_mhd_finetune.py`: three/four outputs, random or
  pretrained initialization, magnetic channel initialization, residual
  rollout, optimizer groups, and paired velocity/B FFT projection.
- `src/losses/physics_informed.py`: vector-potential and direct-B losses,
  plus opt-in KH time-local velocity, magnetic, vorticity, and current losses.
- `scripts/calculate_statistics.py`: optional KH per-time velocity RMS
  ratios, clipping, NPZ output, and memory mapping.
- Training/plotting files: all-time plotting, physical times, corrected
  orientation, and four-channel/derived layouts.
- Canonical three-channel TL config:
  `MHD-World/configs/config_poseidon_mhd_finetune_Re1000_vecpot_TL_bs16_nueta1e3_corrected_100ep.yaml`.
- Canonical four-channel DT config: the best single-Re config above.

KH behavior must remain opt-in and must not change DT defaults.

### Scratch initialization

From `MHD-World-scratch`:

- `run_training.py`: construction without `from_pretrained`.
- `src/neurops/poseidon_mhd_finetune.py`: random initialization controls.
- `src/neurops/poseidon_mhd_re_finetune.py`: compatible constructor.
- Canonical config:
  `configs/config_poseidon_mhd_finetune_Re1000_scratch_physics_nueta1e3_corrected.yaml`.
  The batch-16 variant is an optional comparison, not the reported row.

### Naive multi-regime conditioning

From `MHD-World-Re-naive`:

- `src/neurops/poseidon_mhd_re_input_finetune.py`: broadcast standardized
  log10(Re) and log10(Rm) maps.
- Model-factory registration.
- Metadata-aware training/loss path with per-sample transport coefficients.
- Canonical config:
  `configs/gadi/config_poseidon_mhd_re_input_vecpot_TL_corrected_Re1000_warm_bs1_100ep.yaml`.

Exclude the unrelated spectral-conditioning experiment from the first release.

## Required MHD-World-new source

### Data and normalization

- `src/data/multi_re_neurops_dataset.py`: metadata-aware per-Re
  normalization, autoregressive inputs, physical ranges.
- `src/data/neurops_dataset.py`: physical coordinate ranges.
- `src/normalizations/physics_norm.py`: paired channel groups and optional
  per-Re scales.
- `src/utils/data_utils.py`, `src/utils/batches.py`: metadata plumbing.

### Models and physics

- `src/neurops/poseidon_mhd_finetune.py`: four fields, magnetic
  initialization, optimizer groups, residual rollout, velocity/B projection.
- `src/neurops/poseidon_mhd_re_finetune.py`: FiLM/gated adapters and final
  post-conditioning projection.
- `src/losses/physics_informed.py`: component relative-L2 data/IC terms,
  direct-field MHD residuals, vorticity/current losses, per-sample nu/eta.
- `src/physics/pde_solvers.py`: spectral four-field MHD residuals.

### Training, validation, evaluation

- `src/training/train_neural_operator.py`: full/tiny validation schedules,
  denormalized checkpoint metrics, restart reporting, per-Re plots.
- `src/training/validators/neurops_validator.py`: metadata-aware
  denormalization and component/derived metrics.
- Visualization: direct-B/derived fields, physical time/orientation, spectra.
- `scripts/evaluate_poseidon_magnetic_scot_checkpoint.py`: move to public
  evaluation CLI; default paper reports to held-out test.
- `scripts/convert_dedalus_h5_to_poseidon_bfield_npy.py`: preprocessing.

Canonical configs include the corrected FiLM three-channel ablation, the
four-channel DT multi-Re training chain, and global/per-Re KH templates. Fold
restart YAMLs into one clean config plus generic resume support.

## Required DINOs source

### Feature generation

Diffusion first runs its deterministic conditioner over exact
train/validation/test splits and stores predictions, DNS targets, Re metadata,
and source sample IDs. Consolidate the several PINO, POSEIDON, magnetic, KH,
multi-Re, unseen-Re, and previous-baseline exporters under
`data_generation/diffusion/` into one parameterized generator.

It must expose conditioner family, config/checkpoint, raw-data root, split,
output root, direct/residual target mode, and statistics mode. It must preserve
source IDs and never substitute validation data for test data.

### Data and normalization

- `scripts/calculate_statistics.py`: directory-backed, residual-target, and
  per-Re conditioner/target statistics.
- `src/data/diffusion_dataset.py`: lazy features, residual targets, metadata,
  fractions, per-Re normalization, balanced batches, and eager-mode
  `re_per_sample` handling for unseen Re.
- `src/normalizations/min_max_norm.py`: paired and per-Re paired min-max,
  including log-Re interpolation.
- `src/utils/diffusion_tensor_normalization.py`: separate conditioner/target
  transforms, residual reconstruction, full-field projection.
- Keep divergence validation; flattening is an optional performance utility.

### Diffusion model and training

- `src/diffusion/models/helmholtz_projection.py`: Fourier projection for
  velocity and magnetic pairs, including zero/Nyquist handling.
- `src/diffusion/models/elucidated_diffusion.py`: explicit direct/residual
  modes, full-field projection, metadata, optional derivative losses, padding.
- U-Net: optional Re adapters and configurable padding. Best DT PHASE has no
  diffusion Re conditioning.
- `src/training/train_diffusion.py`: weight-only warm start versus true
  resume, diagnostic intervals, 32-step forwarding, denormalized checkpoint
  selection.
- Trainer and validators: metadata, both normalizers, residual-aware physical
  validation, component/derived and divergence metrics.
- Visualization: direct/residual reconstruction, denormalization, corrected
  orientation, and 32-step plots.
- tFNO/model factory: previous-baseline checkpoint compatibility.

Canonical configs:

- tFNO: `configs/prev_SOTA/tFNO/dinos_repro/config_tfno_Re1000_prev_SOTA_dinos_repro_100ep.yaml`.
- DINO: `prev_SOTA/diffusion/Re1000/configs/config_diffusion_Re1000_prev_SOTA_full_from_official_tfno_epoch100_rerun.yaml`.
- Single-Re DT scOT + diffusion: listed above.
- Multi-Re DT residual diffusion: the epoch-91-conditioner, no-Re-condition,
  per-Re-normalized chain; publish as one clean epoch-0-to-100 config.
- Selected KH configs: listed above.

## Proposed public layout

```text
PHASE/
  README.md
  README_poseidon.md
  README_src_changes.md
  pyproject.toml
  src/phase/
    data/
    normalization/
    physics/
    scot/
    diffusion/
    training/
    evaluation/
    visualization/
  configs/
    previous_baseline/
      tfno/
      dino/
    turbulence/
      single_re/
        scot_re1000.yaml
        diffusion_re1000.yaml
      multi_re/
        scot.yaml
        residual_diffusion.yaml
    kh/
      single_re/
        scot_re1000.yaml
        residual_diffusion_re1000.yaml
      multi_re/
        scot_t0_4.yaml
        residual_diffusion_t0_4.yaml
    ablations/
  scripts/
    prepare_data.py
    generate_features.py
    compute_statistics.py
    train_scot.py
    train_diffusion.py
    evaluate.py
  provenance/
    ablations/
    runs/
  tests/
```

Use one shared scOT implementation and one shared diffusion implementation.
Ablation differences belong in configs/model-factory modes. Keep prior
baselines isolated because DINO uses full-field diffusion while multi-Re PHASE
uses residual diffusion.

## Do not copy as core source

- PBS launchers, machine paths, logs, checkpoints, analysis directories.
- Paper-only plots and one-off forensic scripts.
- Archived sweeps, smoke configs, and epoch-specific restart configs after
  their effective settings are captured.
- KH time-local losses in default DT configs.
- Diffusion Re conditioning or diffusion derivative losses in the reported
  best PHASE config; retain only as explicit ablations.

## Consolidation requirements

1. Base shared scOT code on `MHD-World-new`.
2. Merge scratch initialization from `MHD-World-scratch`.
3. Merge naive conditioning from `MHD-World-Re-naive`.
4. Preserve direct-B prediction, final Helmholtz projection, per-sample PDE
   coefficients, and vorticity/current losses.
5. Import DINOs with visibly separate direct DINO and residual PHASE modes.
6. Keep vector-component normalization paired and metadata-aware.
7. Make warm starts weight-only and reset training state.
8. Use 32 diffusion steps everywhere.
9. Replace absolute paths by data/output roots.
10. Test projection, residual reconstruction, per-Re normalization, balanced
    batches, warm-start reset, split/source IDs, and checkpoint loading.
11. Test KH sub_t, physical time, truncation, time-local loss, and plots.
12. Smoke-test all eight ablations and single-/multi-Re DT and KH chains.

## Items still requiring verification

- Provenance and training epoch of downloaded `tfno_Re1000.pt`.
- KH multi-Re diffusion conditioner epoch naming (85 versus visible 75).
- Historical single-Re DT diffusion mode beyond current code/config evidence;
  absent an explicit flag, recovered code selects direct.
- Isolation of later KH options from DT defaults.
- Exact metric reproduction after consolidation.
- Authorship of individual uncommitted lines.

## Batch 11: residual diffusion

- `src/phase/training/dino_trainer.py`: generalized the locked previous-DINO
  loop to dispatch guarded single- and multi-Re PHASE residual recipes; added
  metadata-aware normalization, physical full-field reconstruction for
  validation, strict weights-only warm starts, and denormalized-relative-L2
  checkpoint selection.
- `src/phase/preprocessing/statistics.py`: added train-only per-Re statistics
  for scOT conditions and `DNS - scOT` residual targets.
- `scripts/generate_scot_diffusion_features.py`: added one single-/multi-Re
  exporter that preserves physical units, deterministic split IDs, and Re
  metadata.
- `configs/turbulence/{single_re,multi_re}/`: added the corrected SR recipe
  and historically faithful reported MR recipe.
- The existing low-level residual reconstruction and full-field Helmholtz
  implementation was retained from the audited DINOs source rather than
  duplicated.
