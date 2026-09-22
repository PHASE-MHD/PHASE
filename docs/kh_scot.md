# Kelvin-Helmholtz scOT training

These recipes train four-channel scOT models for the periodic
Kelvin-Helmholtz dataset. Both learn `[ux,uy,Bx,By]` over `t in [0,5]`, use
every fifth stored frame, and therefore see 51 frames at an effective interval
of 0.1. Both apply Helmholtz projection separately to velocity and magnetic
fields and include direct-B PDE, vorticity, and current losses.

## Single-Re scOT

The `Re=Rm=1000` recipe uses a paired magnetic P99 scale fitted on the
training split, batch size 1, and 100 epochs. Primary-field and current
losses are relative L2 values computed per time slice and then averaged over
time. The vorticity loss retains the historical global space-time relative-L2
reduction.

```bash
export DATA_ROOT=/path/to/kh_four_channel_data
export OUTPUT_ROOT=/path/to/outputs

python scripts/train_scot.py \
  --config configs/kh/single_re/scot_re1000.yaml
```

## Multi-Re scOT

The multi-regime recipe uses one global paired magnetic P99 scale fitted across the
training regimes. It warm-starts model weights only from the selected single-Re KH scOT checkpoint, while epoch,
optimizer, and scheduler state start fresh. Balanced batches contain all ten
regimes through nominal `batch_size=1` and `res_per_batch=10`.

The canonical historical implementation computes primary, vorticity, and
current relative-L2 losses globally over space and time; the time-local options
declared in the legacy config do not alter that reduction. Full validation
runs every five epochs and selects the checkpoint by denormalized relative L2.
A five-sample-per-Re diagnostic runs every epoch for monitoring only.

```bash
python scripts/train_scot.py \
  --config configs/kh/multi_re/scot_t0_5.yaml
```

Both configs contain the exact normalization, optimizer, loss, warm-start, and
checkpoint settings used by the canonical runs.
