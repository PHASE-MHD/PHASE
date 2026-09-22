# Training

- `tfno_trainer.py`: trains the previous tFNO baseline from scratch.
- `scot_trainer.py`: trains the scOT ablations and single- and
  multi-regime turbulence/KH models.
- `dino_trainer.py`: trains full-field DINO and residual PHASE diffusion.

scOT multi-regime warm starts load model weights only; training begins at epoch zero with a new optimizer and scheduler. Diffusion distinguishes the previous full-field DINO objective from PHASE residual learning and projects the full reconstructed velocity and magnetic fields in residual mode.