from functools import partial
import math

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim.lr_scheduler import LambdaLR


@torch.no_grad()
def random_pruning_(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(random_pruning_mask)


@torch.no_grad()
def magnitude_pruning_(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))


def optimizer_reset(
    optimizer,
    *,
    reset_params: list[torch.nn.Parameter],
    optimizer_state_keys: list[str],
    reset_optimizer_on_relora: bool,
    optimizer_random_pruning: float,
    optimizer_magnitude_pruning: float,
):
    """
        optimizer_state_keys: e.g., ["exp_avg", "exp_avg_sq"]
    """
    n_reset_types = (
        int(bool(reset_optimizer_on_relora))
        + int(bool(optimizer_random_pruning))
        + int(bool(optimizer_magnitude_pruning))
    )
    if n_reset_types != 1:
        # logger.warning(f"Got {reset_optimizer_on_relora=}, {optimizer_random_pruning=}, "
        #                f"{optimizer_magnitude_pruning=}")
        raise ValueError(f"Exactly one of reset_optimizer_on_relora, "
                         f"optimizer_random_pruning, optimizer_magnitude_pruning must be True")

    # pruning_fn has to be inplace to work with ZeroRedundancyOptimizer
    if reset_optimizer_on_relora:
        pruning_fn = partial(random_pruning_, prune_ratio=0.999)
    elif optimizer_random_pruning:
        pruning_fn = partial(random_pruning_, prune_ratio=optimizer_random_pruning)
    elif optimizer_magnitude_pruning:
        pruning_fn = partial(magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning)
    else:
        raise ValueError("Unknown pruning type")
        
    n_zeros = 0
    n_total = 0

    optimizer_state = optimizer.state
    if isinstance(optimizer, ZeroRedundancyOptimizer):
        optimizer_state = optimizer.optim.state

    for p in reset_params:
        param_state = optimizer_state[p]
        if len(param_state) == 0: # no state for this param, happens for ZeRo optimizer
            continue
        for key in optimizer_state_keys:
            pruning_fn(param_state[key])  # pruning fn has to be inplace to keep the same keys in the dict
            n_total += param_state[key].numel()
            n_zeros += torch.sum(param_state[key] == 0).item()

    _zeroed = n_zeros / (1e-7 + n_total) * 100
    print(f"Percent of optimizer states zeroed: {_zeroed:.2f}")


def get_cosine_schedule_with_multiple_warmups(
    optimizer,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    adjust_warmup_iters,
    adjust_step,
    min_lr_ratio=0.1,
    last_epoch=-1,
):
    if restart_every is None:
        raise ValueError("restart_every must be specified for cosine_restarts scheduler")

    if num_training_steps % restart_every != 0:
        raise ValueError(f"num_training_steps ({num_training_steps}) must be divisible by restart_every ({restart_every})")

    lr_lambda = partial(
        _get_cosine_schedule_with_multiple_warmups_lambda,
        num_training_steps=num_training_steps,
        first_warmup_steps=first_warmup_steps,
        restart_warmup_steps=restart_warmup_steps,
        restart_every=restart_every,
        min_lr_ratio=min_lr_ratio,
        adjust_step=adjust_step,
        adjust_warmup_iters=adjust_warmup_iters,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def _get_cosine_schedule_with_multiple_warmups_lambda(
    current_step,
    *,
    num_training_steps,
    first_warmup_steps,
    restart_warmup_steps,
    restart_every,
    min_lr_ratio,
    adjust_step,
    adjust_warmup_iters,
):
    """
    Args:
        adjust_step: useful when continuing training from a warmed up checkpoint,
            it allows to sync the resets by reducing the number of steps
            after the first warmup and before the first reset.
            Thus, your ReLoRA resets can be synced with the optimizer resets.
    """
    assert 0 < min_lr_ratio <= 1.0, "min_lr_ratio must be in (0,1]"
    assert restart_every > 0, "restart_every must be positive"
    assert adjust_step + first_warmup_steps <= num_training_steps, "warmup + adjust_step is more than full training steps"
    # assert adjust_step + first_warmup_steps <= restart_every, "the first reset will happen before the warmup is done"

    if current_step < first_warmup_steps:
        progress = float(adjust_step + first_warmup_steps - adjust_warmup_iters) / float(max(1, num_training_steps - adjust_warmup_iters))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(current_step) / float(max(1, first_warmup_steps)) * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay) 

    _current_step = current_step + adjust_step

    restart_step = _current_step % restart_every
    restart_number = _current_step // restart_every

    if restart_step < restart_warmup_steps and current_step >= restart_every:
        # get expected lr multipler at the end of the warmup
        end_of_warmup_progress = (
            float(restart_number * restart_every + restart_warmup_steps - adjust_warmup_iters) /
            float(max(1, num_training_steps - adjust_warmup_iters))
        )

        _cosine_decay = 0.5 * (1.0 + math.cos(math.pi * end_of_warmup_progress))
        warmup_lr_multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * _cosine_decay
    
        return float(restart_step) / float(max(1, restart_warmup_steps)) * warmup_lr_multiplier

    progress = float(_current_step - adjust_warmup_iters) / float(max(1, num_training_steps - adjust_warmup_iters))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
