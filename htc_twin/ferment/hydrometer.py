from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LearnedResidual(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_dim, hidden)
        self.output_layer = nn.Linear(hidden, 1)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.hidden_layer(context))
        return self.output_layer(hidden).squeeze(-1)


class OrdinalReadout(nn.Module):
    def __init__(self, grades: int) -> None:
        super().__init__()
        if grades < 2:
            raise ValueError("ordinal readout needs at least two grades")
        self.grades = grades
        self.raw_thresholds = nn.Parameter(torch.linspace(-1.0, 1.0, grades - 1))
        self.raw_weight = nn.Parameter(torch.tensor(0.0))

    def weight(self) -> torch.Tensor:
        return F.softplus(self.raw_weight) + 1.0e-3

    def thresholds(self) -> torch.Tensor:
        first = self.raw_thresholds[:1]
        gaps = F.softplus(self.raw_thresholds[1:])
        return torch.cat([first, first + torch.cumsum(gaps, dim=0)], dim=0)

    def cumulative(self, latent: torch.Tensor) -> torch.Tensor:
        tau = self.thresholds().unsqueeze(0)
        scaled = self.weight() * latent.unsqueeze(-1)
        return torch.sigmoid(tau - scaled)

    def category_probs(self, latent: torch.Tensor) -> torch.Tensor:
        cum = self.cumulative(latent)
        ones = torch.ones(cum.shape[0], 1, dtype=cum.dtype, device=cum.device)
        extended = torch.cat([cum, ones], dim=1)
        prev = torch.cat([torch.zeros_like(ones), cum], dim=1)
        probs = extended - prev
        return probs.clamp_min(1.0e-8)

    def expected_grade(self, latent: torch.Tensor) -> torch.Tensor:
        probs = self.category_probs(latent)
        levels = torch.arange(self.grades, dtype=probs.dtype, device=probs.device)
        return (probs * levels.unsqueeze(0)).sum(dim=-1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.category_probs(latent)


def severity_latent(cartilage: torch.Tensor, c_max: float, residual: torch.Tensor) -> torch.Tensor:
    return (c_max - cartilage) + residual
