# Kelvin-Helmholtz scOT training

four-channel scOT recipes for the periodic Kelvin-Helmholtz dataset. Both recipes learn (u_x,u_y,B_x,B_y) on
t in [0,5], apply Helmholtz projection to both vector pairs, and use vorticity, and current losses.

## Data layout

Set DATA_ROOT to a directory containing:

    $DATA_ROOT/mhd_Re80_N1000/mhd_data_4channel.npy
    ...
    $DATA_ROOT/mhd_Re4500_N1000/mhd_data_4channel.npy

## Training order

    export DATA_ROOT=/path/to/KH_multiRe
    export OUTPUT_ROOT=/path/to/outputs

    python scripts/train_scot.py \
      --config configs/kh/single_re/scot_re1000.yaml

    python scripts/train_scot.py \
      --config configs/kh/multi_re/scot_t0_5.yaml