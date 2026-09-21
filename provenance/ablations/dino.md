# DINO ablation provenance

## Identity

- Ablation-table record: `paper_draft_plots/ablation_table/DINO.txt`
- Problem: Re=Rm=1000 decaying incompressible MHD turbulence
- Conditioner: externally downloaded tFNO over `[ux, uy, A]`
- Diffusion target: direct/full DNS field, not a residual
- Diffusion warm start: none
- Split: 900 train / 50 validation / 50 held-out test, seed 42
- Temporal subsampling: `sub_t=4`, giving 26 images per trajectory
- Training: 101 epochs (indices 0--100), batch size 64
- Validation: epochs 10, 20, ..., 100; no epoch-zero validation
- Validation and evaluation sampling: 32 EDM steps
- Reported checkpoint: epoch 100, selected by denormalized relative L2
- Training job: `178482355.gadi-pbs`
- Evaluation job: `178575135.gadi-pbs`, held-out test split

## External conditioner

- Legacy path: `DINOs/DINOs_paper_chkpt/tfno_Re1000.pt`
- SHA-256: `a96152ba4dc4b341d9a336c4c619e55835e1776bda8655824f96d25398d59e7d`
- Source folder: `https://drive.google.com/drive/folders/1hTdHoYCdW59gZYDBUgc06TdY7OHGdghi`
- Checkpoint metadata epoch: 234
- Original training job and full history: not independently recoverable

The checkpoint is not copied into the public repository.

## Reported diffusion checkpoint

- Legacy path: `DINOs/prev_SOTA/diffusion/Re1000/checkpoints/diffusion_Re1000_prev_SOTA_full_from_official_tfno_epoch100.pt`
- Size: approximately 4.8 GiB
- SHA-256: `6bcf22161dfe5c96d32dcde6da8d0e3f5b5f582687828b465b66e98fadd7d72b`

The checkpoint is retained only as an external compatibility artifact.

## Frozen source hashes

- EDM: `098f31190f29730c9c1bc2d0940ea6653583bade393facfb267340b1cc32fc18`
- U-Net: `6a7713982b90e7d620b40d2d7a9bead0f2b02ae907c277078849e75e6de80f30`
- attention: `f99eedd81751656759a116749fa7fcdf32e05c759838821cb2334c3ff1a7fed3`
- residual blocks: `19a537f3c17e3145fa8318a2d14ac9678447b9fa0c05ce57101215cd620d0645`
- sampling layers: `a09b3cc6139394f50114fae31fce2197c313690b73d393fb5eac2080ffac4448`
- RMS normalization: `383117606c28416b7a6a269a9330edd662e1729b70de8ba440cbd1e3f305791f`
- diffusion factory: `7229ef7924dbc3add1106db79ce71df6c65a850a4815f174bbcbd096113f0ca9`
- legacy diffusion config: `aad6b0cfacc82a1dca5b3a18f66d2378ce539363391813f278733075b88526cc`
- legacy conditioner config: `dbc40a3e43466f0bf6b86a891dffd45a6a277f8a88eb1c06f724bc9768c7577d`
- legacy feature generator: `7bf0e901cc2249f1a3d382990d497433dd6b5fedfbbb5f32595baab48867545a`

The public trainer removes unrelated multi-Re diagnostics and Gadi-specific
paths but preserves the model constructor, EDM objective, optimizer,
scheduler, validation cadence, sampling settings, and checkpoint criterion.
