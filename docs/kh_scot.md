# Kelvin-Helmholtz scOT training

Batch 12 provides deterministic four-channel scOT recipes for the periodic
Kelvin-Helmholtz dataset. Both recipes learn (u_x,u_y,B_x,B_y) on
t in [0,5], retain 51 frames with sub_t=5, apply Helmholtz projection to
both vector pairs, and use direct-B PDE, vorticity, and current losses.

## Data layout

Set DATA_ROOT to a directory containing:

    $DATA_ROOT/mhd_Re80_N1000/mhd_data_4channel.npy
    ...
    $DATA_ROOT/mhd_Re4500_N1000/mhd_data_4channel.npy

Each array has shape (1000,251,128,128,4) and channel order
(u_x,u_y,B_x,B_y). The seed-42 split is 800 train, 100 validation, and 100
test trajectories per regime. Single-Re training reads the Re=1000 file. The
multi-Re recipe uses all ten Re=Rm values: 80, 200, 400, 650, 1000, 1500,
2050, 2750, 3600, and 4500.

## Normalization and losses

Velocity retains POSEIDON-native fluid normalization. Each magnetic pair shares
one scale so Helmholtz projection remains compatible with normalization.
Single-Re uses 0.0669424514, computed on the true seed-42 training split
after sub_t=5. Multi-Re preserves the selected historical global magnetic
absolute-P99 scale 0.0806614549. That legacy estimator sampled concatenated
|Bx| and |By| values from raw trajectory indices 0:800 at all 251 frames; it
was not computed from the later seed-split histogram routine.

The reported single-Re objective computes spatial relative L2 errors
independently at each time and then averages over time and batch for u_x, u_y, B_x, B_y,
and current; vorticity uses one global space-time relative L2. The reported
multi-Re objective uses global space-time relative L2 for all primary and
derived fields. DT configs retain their existing global-in-time objective.

## Training order

    export DATA_ROOT=/path/to/KH_multiRe
    export OUTPUT_ROOT=/path/to/outputs

    python scripts/train_scot.py \
      --config configs/kh/single_re/scot_re1000.yaml

    python scripts/train_scot.py \
      --config configs/kh/multi_re/scot_t0_5.yaml

The single-Re recipe starts from POSEIDON and writes
$OUTPUT_ROOT/checkpoints/kh_single_re_scot_re1000.pt. The multi-Re recipe
loads only those model weights as a warm start, creates gated Re/Rm adapters,
and starts its optimizer and epoch counter from zero. In the canonical multi-Re
optimizer, the expanded magnetic boundary tensors belong to the pretrained
group at 1e-7; the retained magnetic_output_lr=2e-3 field is inactive when
boundary_group is pretrained.

The multi-Re run performs a cheap, deterministically resampled
five-sample-per-Re diagnostic each epoch. Full validation runs every five
epochs and at the final epoch; only full validation
denormalized relative L2 can replace the best checkpoint. Prediction-panel
generation is intentionally deferred to the unified evaluation tooling rather
than being embedded in the training loop.
