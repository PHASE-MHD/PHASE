"""Standard PyTorch optimizers wrapped with the registry."""

import torch
import torch.optim as optim
import math
from .optimizer_factory import register_optimizer


@register_optimizer("adam")
def create_adam_optimizer(parameters, optimizer_params):
    """Create an Adam optimizer."""
    return optim.Adam(
        parameters,
        lr=optimizer_params.get("lr", 1e-3),
        betas=optimizer_params.get("betas", (0.9, 0.999)),
        eps=optimizer_params.get("eps", 1e-8),
        weight_decay=optimizer_params.get("weight_decay", 0),
        amsgrad=optimizer_params.get("amsgrad", False),
    )


@register_optimizer("adamw")
def create_adamw_optimizer(parameters, optimizer_params):
    """Create an AdamW optimizer with decoupled weight decay."""
    return optim.AdamW(
        parameters,
        lr=optimizer_params.get("lr", 1e-3),
        betas=optimizer_params.get("betas", (0.9, 0.999)),
        eps=optimizer_params.get("eps", 1e-8),
        weight_decay=optimizer_params.get("weight_decay", 1e-2),
        amsgrad=optimizer_params.get("amsgrad", False),
    )


@register_optimizer("sgd")
def create_sgd_optimizer(parameters, optimizer_params):
    """Create an SGD optimizer with optional Nesterov momentum."""
    return optim.SGD(
        parameters,
        lr=optimizer_params.get("lr", 1e-3),
        momentum=optimizer_params.get("momentum", 0.0),
        dampening=optimizer_params.get("dampening", 0),
        weight_decay=optimizer_params.get("weight_decay", 0),
        nesterov=optimizer_params.get("nesterov", False),
    )


@register_optimizer("rmsprop")
def create_rmsprop_optimizer(parameters, optimizer_params):
    """Create an RMSprop optimizer."""
    return optim.RMSprop(
        parameters,
        lr=optimizer_params.get("lr", 1e-2),
        alpha=optimizer_params.get("alpha", 0.99),
        eps=optimizer_params.get("eps", 1e-8),
        weight_decay=optimizer_params.get("weight_decay", 0),
        momentum=optimizer_params.get("momentum", 0),
        centered=optimizer_params.get("centered", False),
    )


@register_optimizer("lbfgs")
def create_lbfgs_optimizer(parameters, optimizer_params):
    """Create an L-BFGS optimizer."""
    return optim.LBFGS(
        parameters,
        lr=optimizer_params.get("lr", 1),
        max_iter=optimizer_params.get("max_iter", 20),
        max_eval=optimizer_params.get("max_eval", None),
        tolerance_grad=optimizer_params.get("tolerance_grad", 1e-7),
        tolerance_change=optimizer_params.get("tolerance_change", 1e-9),
        history_size=optimizer_params.get("history_size", 100),
        line_search_fn=optimizer_params.get("line_search_fn", None),
    )


@register_optimizer("adagrad")
def create_adagrad_optimizer(parameters, optimizer_params):
    """Create an Adagrad optimizer."""
    return optim.Adagrad(
        parameters,
        lr=optimizer_params.get("lr", 1e-2),
        lr_decay=optimizer_params.get("lr_decay", 0),
        weight_decay=optimizer_params.get("weight_decay", 0),
        initial_accumulator_value=optimizer_params.get("initial_accumulator_value", 0),
        eps=optimizer_params.get("eps", 1e-10),
    )


@register_optimizer("adadelta")
def create_adadelta_optimizer(parameters, optimizer_params):
    """Create an Adadelta optimizer."""
    return optim.Adadelta(
        parameters,
        lr=optimizer_params.get("lr", 1.0),
        rho=optimizer_params.get("rho", 0.9),
        eps=optimizer_params.get("eps", 1e-6),
        weight_decay=optimizer_params.get("weight_decay", 0),
    )


# For Lion optimizer, which is not in standard PyTorch but can be implemented
@register_optimizer("lion")
def create_lion_optimizer(parameters, optimizer_params):
    """
    Create a Lion optimizer (Learning Rate over Independent Optimizer Norms).

    Note: Requires installing the lion-pytorch package:
    pip install lion-pytorch

    Or you can use the implementation below.
    """
    try:
        from lion_pytorch import Lion

        return Lion(
            parameters,
            lr=optimizer_params.get("lr", 1e-4),
            betas=optimizer_params.get("betas", (0.9, 0.99)),
            weight_decay=optimizer_params.get("weight_decay", 1e-2),
        )
    except ImportError:
        # Simple Lion implementation if the package is not available
        class Lion(optim.Optimizer):
            def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
                defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
                super().__init__(params, defaults)

            @torch.no_grad()
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                for group in self.param_groups:
                    for p in group["params"]:
                        if p.grad is None:
                            continue

                        # Perform stepweight decay
                        p.data.mul_(1 - group["lr"] * group["weight_decay"])

                        grad = p.grad
                        state = self.state[p]

                        # State initialization
                        if len(state) == 0:
                            state["exp_avg"] = torch.zeros_like(p)

                        exp_avg = state["exp_avg"]
                        beta1, beta2 = group["betas"]

                        # Update moving average
                        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                        # Update weights using Lion rule
                        update = exp_avg.sign()
                        p.add_(update, alpha=-group["lr"])

                return loss

        return Lion(
            parameters,
            lr=optimizer_params.get("lr", 1e-4),
            betas=optimizer_params.get("betas", (0.9, 0.99)),
            weight_decay=optimizer_params.get("weight_decay", 1e-2),
        )


# For RAdam, which is not in standard PyTorch
@register_optimizer("radam")
def create_radam_optimizer(parameters, optimizer_params):
    """
    Create a RAdam optimizer (Rectified Adam).

    Note: For the full implementation, consider installing a package like 'torch-optimizer'.
    pip install torch-optimizer
    """
    try:
        from torch_optimizer import RAdam

        return RAdam(
            parameters,
            lr=optimizer_params.get("lr", 1e-3),
            betas=optimizer_params.get("betas", (0.9, 0.999)),
            eps=optimizer_params.get("eps", 1e-8),
            weight_decay=optimizer_params.get("weight_decay", 0),
        )
    except ImportError:
        print(
            "Warning: RAdam requested but torch-optimizer not installed. Falling back to Adam."
        )
        print("Please install torch-optimizer: pip install torch-optimizer")
        return create_adam_optimizer(parameters, optimizer_params)


# For AdaBelief, which is not in standard PyTorch
@register_optimizer("adabelief")
def create_adabelief_optimizer(parameters, optimizer_params):
    """
    Create an AdaBelief optimizer.

    Note: Requires installing the adabelief-pytorch package:
    pip install adabelief-pytorch
    """
    try:
        from adabelief_pytorch import AdaBelief

        return AdaBelief(
            parameters,
            lr=optimizer_params.get("lr", 1e-3),
            betas=optimizer_params.get("betas", (0.9, 0.999)),
            eps=optimizer_params.get("eps", 1e-8),
            weight_decay=optimizer_params.get("weight_decay", 0),
            weight_decouple=optimizer_params.get("weight_decouple", False),
            rectify=optimizer_params.get("rectify", False),
        )
    except ImportError:
        print(
            "Warning: AdaBelief requested but package not installed. Falling back to Adam."
        )
        print("Please install AdaBelief: pip install adabelief-pytorch")
        return create_adam_optimizer(parameters, optimizer_params)
