from __future__ import annotations

import torch
from torch.nn import functional as F


def dt_cross_entropy(
    logits: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_actions = actions.reshape(-1).to(torch.long)
    losses = F.cross_entropy(flat_logits, flat_actions, reduction="none")
    flat_mask = mask.reshape(-1).to(losses.dtype)
    denom = flat_mask.sum().clamp_min(1.0)
    return (losses * flat_mask).sum() / denom
