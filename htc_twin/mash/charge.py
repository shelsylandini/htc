from __future__ import annotations

import torch

from htc_twin.recipe import ActionGrid


def static_guideline(grid: ActionGrid, sharpness: float = 3.0) -> torch.Tensor:
    loads = grid.loads()
    preference = -sharpness * (loads - loads.min())
    return torch.softmax(preference, dim=0)


def sample_actions(probs: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    return torch.multinomial(probs, count, replacement=True, generator=generator)


def behaviour_log_prob(probs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return torch.log(probs.gather(0, actions.reshape(-1).to(torch.long)).clamp_min(1.0e-12))
