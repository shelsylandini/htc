from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class LoggedDataset:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminals: torch.Tensor
    timesteps: torch.Tensor
    strata: torch.Tensor
    baseline: torch.Tensor
    cartilage: torch.Tensor
    next_cartilage: torch.Tensor
    grades: torch.Tensor
    behaviour: torch.Tensor

    @property
    def n_patients(self) -> int:
        return int(self.states.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.states.shape[1])

    @property
    def state_dim(self) -> int:
        return int(self.states.shape[2])
