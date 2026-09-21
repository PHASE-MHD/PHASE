# tFNO ablation provenance

## Identity

- Ablation-table record: `paper_draft_plots/ablation_table/tFNO.txt`
- Meaning: previous-study tFNO reproduction at `Re=Rm=1000`
- Fields: `[ux, uy, A]`
- Warm start: none; the legacy `load_ckpt` value is empty
- Split: 900 train / 50 validation / 50 held-out test, seed 42
- Temporal subsampling: `sub_t=4`
- Training: 100 epochs (indices 0--99), batch size 1
- Validation: every epoch
- Gradient clipping: maximum norm 1.0
- Best checkpoint: epoch 99, selected by model validation loss
- Legacy job: `178392259`

## Frozen legacy sources

- Config: `DINOs/configs/prev_SOTA/tFNO/dinos_repro/config_tfno_Re1000_prev_SOTA_dinos_repro_100ep.yaml`
- Launcher: `DINOs/scripts/prev_SOTA/tFNO/dinos_repro/job_train_tfno_Re1000_prev_SOTA_dinos_repro_100ep_gpuhopper.sh`
- Checkpoint: `DINOs/checkpoints/prev_SOTA/tFNO/dinos_repro/tFNO_Re1000_prev_SOTA_dinos_repro_100ep.pt`
- Evaluation: `DINOs/prev_SOTA/tFNO/dinos_repro/evaluation/Re1000/tfno_vecpot_derived_evaluation.txt`

## Source hashes

- config: `64955306e05cbcad5c010a4214f5b41b552cc4443fd640c000c7460a63f15249`
- tFNO: `5917c1ae0c855ec141d3eb246b1ca92085d69f69ac1a62c86221c2fe88a86af8`
- factorized spectral layer: `3fab3b732e93cfe43f1369eb284d3bb81c97d7b40c6d64708b59a6d0aa52c8fe`
- vector-potential loss: `3be086210baaa23aa02e9fc26d089d7a739481bbd28e29fbba26005adeb64d3d`
- PDE residuals: `a7201eb66ab497851e2d16917c9492af2544cdaac7179598e50d3a662070ef7a`
- constraints: `58dc28419278b46e5f44771f5e9c452aa2dad89971f22441a2796ea354e6b0bb`
- Fourier utilities: `2e66a2fe897639756d300c56d266556683d2eea1bf48bd9d2fb051b2f6a7990e`
- normalization utilities: `12737e6a302ee546fd14df4e39877e195c071b323718d9d65bd322e13da9094f`

The public files use package-relative imports and environment-variable paths.
Those portability changes do not alter the model or objective. The model
factory explicitly registers only tFNO, avoiding imports of unused FNO and UNO
backends.
