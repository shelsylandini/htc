from __future__ import annotations

from typing import TypedDict

import torch

from htc_twin.mash.cohort import LoggedDataset


class TransitionBatch(TypedDict):
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminals: torch.Tensor
    grades: torch.Tensor


class SequenceBatch(TypedDict):
    returns: torch.Tensor
    states: torch.Tensor
    actions: torch.Tensor
    timesteps: torch.Tensor
    mask: torch.Tensor


class TransitionLoader:
    def __init__(self, dataset: LoggedDataset) -> None:
        self.states = dataset.states.reshape(-1, dataset.state_dim)
        self.actions = dataset.actions.reshape(-1)
        self.rewards = dataset.rewards.reshape(-1)
        self.next_states = dataset.next_states.reshape(-1, dataset.state_dim)
        self.terminals = dataset.terminals.reshape(-1)
        self.grades = dataset.grades.reshape(-1)
        self.size = self.states.shape[0]

    def sample(self, batch_size: int, generator: torch.Generator) -> TransitionBatch:
        index = torch.randint(0, self.size, (batch_size,), generator=generator)
        return TransitionBatch(
            states=self.states[index],
            actions=self.actions[index],
            rewards=self.rewards[index],
            next_states=self.next_states[index],
            terminals=self.terminals[index],
            grades=self.grades[index],
        )


def build_sequences(dataset: LoggedDataset, returns: torch.Tensor) -> SequenceBatch:
    mask = torch.ones_like(dataset.actions, dtype=torch.float32)
    return SequenceBatch(
        returns=returns,
        states=dataset.states,
        actions=dataset.actions,
        timesteps=dataset.timesteps,
        mask=mask,
    )


def sample_sequences(
    batch: SequenceBatch, batch_size: int, generator: torch.Generator
) -> SequenceBatch:
    total = batch["actions"].shape[0]
    index = torch.randint(0, total, (batch_size,), generator=generator)
    return SequenceBatch(
        returns=batch["returns"][index],
        states=batch["states"][index],
        actions=batch["actions"][index],
        timesteps=batch["timesteps"][index],
        mask=batch["mask"][index],
    )
