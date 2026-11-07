from __future__ import annotations

import torch


def default_cutpoints() -> torch.Tensor:
    return torch.tensor([0.15, 0.35, 0.55, 0.75])


def kl_grades(cartilage: torch.Tensor, c_max: float, cutpoints: torch.Tensor) -> torch.Tensor:
    damage = (c_max - cartilage).clamp_min(0.0)
    grade = torch.zeros_like(cartilage, dtype=torch.long)
    for cut in cutpoints:
        grade = grade + (damage > cut).to(torch.long)
    return grade


def benefit_index(cartilage: torch.Tensor, inflammation: torch.Tensor) -> torch.Tensor:
    return 80.0 * cartilage - 20.0 * inflammation
