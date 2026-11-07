from __future__ import annotations

from typing import NamedTuple

import torch


class Qubo(NamedTuple):
    matrix: torch.Tensor
    constant: float


def symmetrise(mat: torch.Tensor) -> torch.Tensor:
    return 0.5 * (mat + mat.transpose(-1, -2))


def build_qubo(weights: torch.Tensor, objectives: torch.Tensor, penalty: float) -> Qubo:
    n = objectives.shape[-1]
    cost = torch.zeros(n, n, dtype=objectives.dtype)
    for index in range(weights.shape[0]):
        cost = cost + weights[index] * symmetrise(objectives[index])
    ones = torch.ones(n, n, dtype=objectives.dtype)
    matrix = -cost + penalty * ones
    matrix = matrix - 2.0 * penalty * torch.eye(n, dtype=objectives.dtype)
    return Qubo(matrix=matrix, constant=float(penalty))


def qubo_energy(bits: torch.Tensor, qubo: Qubo) -> torch.Tensor:
    quad = torch.einsum("...i,ij,...j->...", bits, qubo.matrix, bits)
    return quad + qubo.constant


def objective_vector(bits: torch.Tensor, objectives: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            torch.einsum("...i,ij,...j->...", bits, symmetrise(objectives[k]), bits)
            for k in range(objectives.shape[0])
        ],
        dim=-1,
    )


def one_hot_feasible(bits: torch.Tensor) -> torch.Tensor:
    return bits.sum(dim=-1) == 1.0
