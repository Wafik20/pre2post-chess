# training/optim_sched.py
from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple, Optional, Callable
import math
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim import AdamW, Adam, SGD, RMSprop, Adagrad
from torch.optim.lr_scheduler import (
    _LRScheduler, CosineAnnealingLR, StepLR, ExponentialLR, ReduceLROnPlateau,
    ConstantLR, LinearLR, LambdaLR
)

def _param_groups(model: nn.Module, weight_decay: float, *, exclude_bias_norm: bool) -> Iterable[dict]:
    if not exclude_bias_norm:
        return [{"params": [p for p in model.parameters() if p.requires_grad],
                 "weight_decay": weight_decay}]
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.endswith(".bias") or ("norm" in n.lower()) or ("ln" in n.lower()) or ("layernorm" in n.lower()):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = []
    if decay: groups.append({"params": decay, "weight_decay": weight_decay})
    if no_decay: groups.append({"params": no_decay, "weight_decay": 0.0})
    return groups

def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> Optimizer:
    name = (cfg.get("name") or "adamw").lower()
    lr = float(cfg.get("lr"))
    if lr is None:
        raise ValueError("training.optimizer.lr is required")
    wd = float(cfg.get("weight_decay", 0.1))
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = float(cfg.get("eps", 1e-8))
    momentum = float(cfg.get("momentum", 0.9))
    alpha = float(cfg.get("alpha", 0.99))  # RMSprop
    exclude_bias_norm = bool(cfg.get("exclude_bias_and_norm", True))
    fused_flag = bool(cfg.get("fused", False))
    nesterov = bool(cfg.get("nesterov", False))

    params = _param_groups(model, wd, exclude_bias_norm=exclude_bias_norm)

    if name == "adamw":
        kwargs = {"lr": lr, "weight_decay": wd, "betas": betas, "eps": eps}
        if fused_flag and torch.__version__ >= "2.0":
            kwargs["fused"] = True
        return AdamW(params, **kwargs)
    if name == "adam":
        kwargs = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": wd}
        if fused_flag and torch.__version__ >= "2.0":
            kwargs["fused"] = True
        return Adam(params, **kwargs)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd)
    if name == "rmsprop":
        return RMSprop(params, lr=lr, alpha=alpha, eps=eps, momentum=momentum, weight_decay=wd, centered=False)
    if name == "adagrad":
        return Adagrad(params, lr=lr, eps=eps, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")

# ---- Schedulers ----

def _total_steps(
    *,
    total_steps: Optional[int],
    steps_per_epoch: Optional[int],
    epochs: Optional[int],
    grad_accum_steps: int = 1,
) -> int:
    if total_steps is not None:
        return int(total_steps)

    if steps_per_epoch is None or epochs is None:
        raise ValueError("Need total_steps or (steps_per_epoch and epochs) for scheduler")

    grad_accum_steps = max(1, int(grad_accum_steps))

    # steps_per_epoch is often dataloader iterations; convert to optimizer steps.
    opt_steps_per_epoch = int(math.ceil(int(steps_per_epoch) / grad_accum_steps))
    return max(1, opt_steps_per_epoch * int(epochs))

def _warmup_lambda(warmup_steps: int, after_fn: Callable[[int], float]) -> Callable[[int], float]:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
        return after_fn(step - warmup_steps if warmup_steps > 0 else step)
    return lr_lambda

def build_scheduler(
    optimizer: Optimizer,
    cfg: Dict[str, Any],
    *,
    steps_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
    total_steps: Optional[int] = None,
    grad_accum_steps: int = 1,
) -> _LRScheduler:
    name = (cfg.get("name") or "cosine").lower()
    eta_min = float(cfg.get("eta_min", 0.0))
    warmup_steps = cfg.get("warmup_steps", None)
    decay_steps = int(cfg.get("decay_steps", 0))  # for step/poly
    gamma = float(cfg.get("gamma", 0.1))          # for step/exp
    plateau_metric = cfg.get("plateau_metric", "loss")
    plateau_patience = int(cfg.get("plateau_patience", 5))
    plateau_factor = float(cfg.get("plateau_factor", 0.5))
    last_epoch = int(cfg.get("last_epoch", -1))

    T = _total_steps(
        total_steps=total_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        grad_accum_steps=grad_accum_steps,
    )
    if warmup_steps is None:
        warmup_steps = int(T*0.01) # default warmup steps is 10% of total steps
    else:
        warmup_steps = int(warmup_steps)

    if name == "constant":
        if warmup_steps > 0:
            return LambdaLR(optimizer, _warmup_lambda(warmup_steps, after_fn=lambda s: 1.0), last_epoch=last_epoch)
        return ConstantLR(optimizer, factor=1.0, total_iters=1, last_epoch=last_epoch)

    if name == "linear":
        # linear decay to eta_min with optional warmup
        # Convert eta_min from absolute LR to ratio for LambdaLR
        base_lr = optimizer.param_groups[0]['lr']
        eta_min_ratio = eta_min / base_lr if base_lr > 0 else 0.0
        def after(step: int) -> float:
            remain = max(0, T - warmup_steps)
            if remain == 0: return 1.0
            progress = min(step, remain) / float(remain)
            # Linear decay from 1.0 to eta_min_ratio
            return max(eta_min_ratio, 1.0 - progress * (1.0 - eta_min_ratio))
        return LambdaLR(optimizer, _warmup_lambda(warmup_steps, after), last_epoch=last_epoch)

    if name in ("cosine", "cosine_warmup"):
        if warmup_steps <= 0 and name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=T, eta_min=eta_min, last_epoch=last_epoch)
        # Convert eta_min from absolute LR to ratio for LambdaLR
        base_lr = optimizer.param_groups[0]['lr']
        eta_min_ratio = eta_min / base_lr if base_lr > 0 else 0.0
        def after(step: int) -> float:
            remain = max(1, T - warmup_steps)
            x = min(step, remain) / float(remain)
            # Cosine annealing from 1.0 to eta_min_ratio
            return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * x))
        return LambdaLR(optimizer, _warmup_lambda(warmup_steps, after), last_epoch=last_epoch)

    if name == "step":
        # step every `decay_steps` by factor gamma, with optional warmup
        if decay_steps <= 0:
            raise ValueError("scheduler.step requires decay_steps>0")
        def after(step: int) -> float:
            k = step // decay_steps
            return gamma ** k
        return LambdaLR(optimizer, _warmup_lambda(warmup_steps, after), last_epoch=last_epoch)

    if name == "exponential":
        return ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)

    if name == "plateau":
        # caller must call scheduler.step(metric) each log step
        return ReduceLROnPlateau(optimizer, mode="min" if plateau_metric == "loss" else "max",
                                 factor=plateau_factor, patience=plateau_patience, verbose=False)

    raise ValueError(f"Unsupported scheduler: {name}")
