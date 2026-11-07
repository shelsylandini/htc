from __future__ import annotations

import torch
from torch import nn


class ConservativeQNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions)
        self.n_actions = n_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(state))

    def gather(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = self.forward(state)
        return values.gather(1, action.view(-1, 1).to(torch.long)).squeeze(1)

    def greedy(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state).argmax(dim=-1)
