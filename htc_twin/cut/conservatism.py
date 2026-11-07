from __future__ import annotations

from typing import NamedTuple

import torch
from torch.nn import functional as F


class CqlTerms(NamedTuple):
    total: torch.Tensor
    temporal_difference: torch.Tensor
    conservative: torch.Tensor


def cql_objective(
    q_all: torch.Tensor,
    q_data: torch.Tensor,
    td_target: torch.Tensor,
    alpha: float,
) -> CqlTerms:
    temporal_difference = F.mse_loss(q_data, td_target.detach())
    conservative = (torch.logsumexp(q_all, dim=1) - q_data).mean()
    total = temporal_difference + alpha * conservative
    return CqlTerms(total=total, temporal_difference=temporal_difference, conservative=conservative)


def td_target(
    reward: torch.Tensor,
    gamma: float,
    next_value: torch.Tensor,
    terminal: torch.Tensor,
) -> torch.Tensor:
    return reward + gamma * (1.0 - terminal) * next_value
