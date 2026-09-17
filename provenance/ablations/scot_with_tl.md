# scOT with POSEIDON transfer learning

## Reported result

This entry corresponds to `+ Transfer learning` in the ablation table. It is
a three-channel `[u_x,u_y,A]` scOT trained at `Re=Rm=1000` with batch size
16 and pretrained `camlab-ethz/Poseidon-T` velocity representations.

## Legacy sources

- Repository: `MHD-World`
- Config:
  `configs/config_poseidon_mhd_finetune_Re1000_vecpot_TL_bs16_nueta1e3_corrected_100ep.yaml`
- Model: `src/neurops/poseidon_mhd_finetune.py`
- Loss: `src/losses/physics_informed.py`
- Training job: `178884733.gadi-pbs`
- Selected checkpoint epoch: 98
- Selection metric: normalized validation loss
- Evaluation split: held-out test split, 100 trajectories

The successful run completed epochs 0--99 in 04:05:31 on one H200. At epoch
98, the normalized validation loss was `4.343096699`, denormalized validation
relative L2 was `3.263769884e-2`, and denormalized validation MSE was
`3.125798477e-5`.

Held-out-test relative L2 errors were:

```text
ux      3.247708e-2
uy      3.417598e-2
Bx      2.487769e-1
By      2.513671e-1
omega   1.345443e-1
J       1.430413e0
```

The seed-42 split-index SHA-256 digests, shared with the no-transfer ablation,
are:

```text
train 0641723ad566748f655b65fff2db399e6be202f94cb7b8a61c4f53c17d7c281e
val   17bfabf318edf05ff63c4994cc37ddc4b26d2ac3a8ce61467533fed9e3fd4d1b
test  0b391decce865a16af4d2598872625588730b2a5a4257bd04160f88c4098b5a9
```

## Legacy artifact hashes

```text
config     fcf2a26fddd7582c9b7a9720c3d51ab57c8a06bf3d1aeaa4944ef99c96365288
model      bf87005e265eceddd106175998d0673d346a5f3bb324b9e19cfc4743e23ba901
loss       2acf4e0af52ef5345664e3b875bf90c4a729086e589f9e16742aca16e7ec1cc4
checkpoint b298e6762f7322c5806668224683e9f95a200359f9781c8a88bb8262800c8d8d
```

## Public mapping

- Config: `configs/ablations/scot_with_tl/re1000.yaml`
- Shared model wrapper: `src/phase/models/scot_mhd.py`
- Shared vector-potential loss: `src/phase/losses/physics_informed.py`
- Shared guarded trainer: `src/phase/training/scot_trainer.py`
- CLI: `scripts/train_scot.py`

The later legacy model file also contains four-channel and Helmholtz features,
but this config activates only the original three-channel path. Those inactive
features are not part of this ablation.

## Compatibility verification

The public wrapper strictly loaded every epoch-98 checkpoint key and contains
20,776,206 parameters. For identical input and checkpoint state, the public
and legacy wrappers produced bit-identical output
(`max_abs_diff=0`). The two AdamW groups exactly matched the checkpoint:
839 copied-parameter tensors at `lr=5e-6, wd=1e-2`, and five expanded-boundary
parameter tensors at `lr=5e-4, wd=0`. Total loss, prediction gradients, and
all 19 logged loss components were bit-identical.

The public config hash at Batch 7 validation was
`771887c12775af0498751705755b7c478ec0d64230107c67e7d8efec2df0d840`.

## Inherited limitation

The legacy config did not pin a Hugging Face revision for
`camlab-ethz/Poseidon-T`. The local cache currently resolves it to
`ec976ed5d25883ec9db4e486ebbeeefa9e08303b`, but no surviving run artifact
proves that this was the revision fetched for the historical training run.

The dataset permutation is deterministic, but the legacy training launcher did
not set a PyTorch seed for shuffled batches. A fresh run therefore reproduces
the architecture, split, objective, optimizer, and selection semantics, but
not necessarily the reported weights bit-for-bit.
