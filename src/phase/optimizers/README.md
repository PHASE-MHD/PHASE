# Optimizers

- `optimizer_factory.py`: registers and constructs optimizers.
- `standard_optimizers.py`: provides the PyTorch optimizer wrappers used by the factory. All canonical PHASE and baseline recipes use AdamW.
- `scheduler_factory.py`: registers and constructs learning-rate schedulers.
- `standard_schedulers.py`: provides constant, cosine, multistep, and validation-plateau scheduling.

scOT recipes use constant learning rates, with model-defined parameter groups for pretrained and newly initialized parameters. PHASE diffusion and DINO use cosine scheduling, while the tFNO baseline uses validation-loss-based plateau scheduling.