# scOT without transfer learning

## Reported result

This provenance entry corresponds to `scOT (u,A)` in the ablation table. The
model was trained from random initialization at `Re=Rm=1000` with batch size
1. It must not be confused with the later corrected batch-size-16 comparison.

## Legacy sources

- Repository: `MHD-World-scratch`
- Config: `configs/config_poseidon_mhd_finetune_Re1000_scratch_physics_nueta1e3_corrected.yaml`
- Model: `src/neurops/poseidon_mhd_finetune.py`
- Loss: `src/losses/physics_informed.py`
- Training job: `178884285.gadi-pbs`
- Selected checkpoint epoch: 95
- Selection metric: normalized validation loss
- Evaluation split used for the final ablation record: held-out test split

The successful run completed epochs 0--99 in 37:43:32 on one H200. At the
selected epoch, the normalized validation loss was `2.2108579e1` and the
denormalized validation relative L2 was `2.69829e-1`.

Held-out-test relative L2 errors over 100 trajectories were:

```text
ux      2.551481e-1
uy      2.662414e-1
Bx      6.798099e-1
By      6.717552e-1
omega   5.302343e-1
J       3.009337e0
```

The NumPy seed-42 split-index SHA-256 digests are:

```text
train 0641723ad566748f655b65fff2db399e6be202f94cb7b8a61c4f53c17d7c281e
val   17bfabf318edf05ff63c4994cc37ddc4b26d2ac3a8ce61467533fed9e3fd4d1b
test  0b391decce865a16af4d2598872625588730b2a5a4257bd04160f88c4098b5a9
```

## Artifact hashes

```text
config     d31877e2bff6e3c7575d60e07ba669c033b4fe4a591cc877dc27b696c5e536aa
model      62825257b2c70f66023d06e4a2a52a84793d9c5aa6b96af3f7989a3255893726
loss       a1ee2b01b7295821f645bf46755e73ac95e909a6cce71e1c7f41db541f0eb4ca
checkpoint 15dfe9b88cf6a93589b14ec6425f2ba82c563c6dc0da716447362523cfc33648
```

The exact corrected legacy config was untracked in the legacy working tree;
its SHA-256 hash above is therefore the immutable provenance identifier. The
model source last appears in legacy commit
`7ea3e41243aeccf10fba030d84b02e319fad871e`.

## External architecture

The scOT implementation is supplied by the external POSEIDON repository at
commit `b8fa28f59bd7f7673323f28d11a12c6f3a215c61`. This ablation uses its
architecture only: `poseidon_model: null` and
`load_pretrained_poseidon: false` ensure no pretrained weights are loaded.
Consequently, the YAML magnetic input/output initialization labels are
inactive; those options only initialize channels added to pretrained weights.
The full scratch model uses the standard random scOT initialization.

## Public mapping

- Config: `configs/ablations/scot_without_tl/re1000.yaml`
- Model wrapper: `src/phase/models/scot_mhd.py`
- Loss: `src/phase/losses/physics_informed.py`
- Trainer: `src/phase/training/scot_trainer.py`
- CLI: `scripts/train_scot.py`

Strict loading of the epoch-95 checkpoint was verified with all parameter keys
matching. The public model contains 11,212,890 parameters.
