# Data

Dataset and dataloader implementations for deterministic operators and diffusion.

- `neurops_dataset.py`: single-regime MHD trajectories and train/validation/test loaders.
- `multi_re_neurops_dataset.py`: multi-regime trajectories, Re/Rm metadata, and balanced regime sampling.
- `diffusion_dataset.py`: conditioner/DNS feature stores, residual targets, normalization, and diffusion loaders.
- `__init__.py`: public dataset exports.

Deterministic trajectories use `[sample,channel,time,x,y]` internally. Model
inputs combine coordinates with the repeated initial condition, while targets
contain the complete trajectory. Diffusion features store conditioner
predictions and matching DNS trajectories; PHASE forms `DNS - conditioner`
targets when residual mode is enabled.

Dataset splits are deterministic. Normalization statistics must be fitted only
on the training split, and multi-regime loaders preserve Re/Rm and source
sample IDs for conditioning and evaluation.
