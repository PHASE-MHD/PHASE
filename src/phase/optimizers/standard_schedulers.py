"""Standard PyTorch learning rate schedulers wrapped with the registry."""

import torch.optim as optim
from .scheduler_factory import register_scheduler


@register_scheduler("dummy")
def create_dummy_scheduler(optimizer, optimizer_params):
    """Create a dummy scheduler that doesn't change the learning rate."""
    return optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)


@register_scheduler("multistep")
def create_multistep_scheduler(optimizer, optimizer_params):
    """Create a MultiStepLR scheduler."""
    return optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=optimizer_params.get("milestones", [50, 100, 150]),
        gamma=optimizer_params.get("gamma", 0.5),
    )


@register_scheduler("cosine")
def create_cosine_scheduler(optimizer, optimizer_params):
    """Create a CosineAnnealingLR scheduler."""
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=optimizer_params.get("T_max", 100),
        eta_min=optimizer_params.get("eta_min", 0),
    )


@register_scheduler("reducelr")
def create_reduce_on_plateau_scheduler(optimizer, optimizer_params):
    """Create a ReduceLROnPlateau scheduler."""
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=optimizer_params.get("mode", "min"),
        factor=optimizer_params.get("factor", 0.5),
        patience=optimizer_params.get("patience", 10),
        threshold=optimizer_params.get("threshold", 1e-4),
        threshold_mode=optimizer_params.get("threshold_mode", "rel"),
        cooldown=optimizer_params.get("cooldown", 0),
        min_lr=optimizer_params.get("min_lr", 0),
        eps=optimizer_params.get("eps", 1e-8),
    )
