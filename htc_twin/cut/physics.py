from __future__ import annotations

import torch


def physics_penalty(trajectory: torch.Tensor, c_max: float) -> torch.Tensor:
    cartilage = trajectory[..., 0]
    inflammation = trajectory[..., 1]
    over = torch.clamp_min(cartilage - c_max, 0.0)
    under = torch.clamp_min(-cartilage, 0.0)
    negative_inflammation = torch.clamp_min(-inflammation, 0.0)
    return (over.pow(2) + under.pow(2) + negative_inflammation.pow(2)).mean()
