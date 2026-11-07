from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


def build_optimiser(
    parameters: Iterable[nn.Parameter], lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)


def warmup_cosine(step: int, warmup: int, total: int) -> float:
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    if total <= warmup:
        return 1.0
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(
    optimiser: torch.optim.Optimizer, warmup: int, total: int
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimiser, lr_lambda=lambda step: warmup_cosine(step, warmup, total)
    )
