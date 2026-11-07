from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from htc_twin.blend.alembic import decode_qaoa, optimise_qaoa
from htc_twin.blend.qubo import build_qubo, objective_vector
from htc_twin.blend.vatting import brute_force_qubo
from htc_twin.nose.hypervolume import normalized_hypervolume
from htc_twin.recipe import QuantumConfig


def weight_grid(objectives: int, divisions: int) -> list[tuple[float, ...]]:
    out: list[tuple[float, ...]] = []

    def recurse(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 1:
            composition = prefix + [remaining]
            out.append(tuple(value / divisions for value in composition))
            return
        for value in range(remaining + 1):
            recurse(prefix + [value], remaining - value, slots - 1)

    recurse([], divisions, objectives)
    return out


def pareto_front(points: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    count = points.shape[0]
    keep = np.ones(count, dtype=bool)
    for i in range(count):
        if not keep[i]:
            continue
        for j in range(count):
            if i == j:
                continue
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                keep[i] = False
                break
    return keep


class SweepResult(NamedTuple):
    quantum_front: npt.NDArray[np.float64]
    classical_front: npt.NDArray[np.float64]
    quantum_hypervolume: float
    classical_hypervolume: float

    @property
    def advantage(self) -> float:
        return self.quantum_hypervolume - self.classical_hypervolume


def _selected_objective(bits: torch.Tensor, objectives: torch.Tensor) -> npt.NDArray[np.float64]:
    vector = objective_vector(bits, objectives)
    return vector.detach().cpu().numpy().astype(np.float64)


def sweep(objectives: torch.Tensor, cfg: QuantumConfig) -> SweepResult:
    n = objectives.shape[-1]
    grid = weight_grid(objectives.shape[0], cfg.grid)
    quantum_points: list[npt.NDArray[np.float64]] = []
    classical_points: list[npt.NDArray[np.float64]] = []
    for index, weights in enumerate(grid):
        weight_tensor = torch.tensor(weights, dtype=objectives.dtype)
        qubo = build_qubo(weight_tensor, objectives, cfg.penalty)
        result = optimise_qaoa(
            qubo, n, cfg.depth, cfg.opt_steps, cfg.opt_lr, cfg.seed + index, cfg.restarts
        )
        q_bits = decode_qaoa(result.probabilities, qubo, n, cfg.shots, cfg.seed + index)
        quantum_points.append(_selected_objective(q_bits, objectives))
        c_bits = brute_force_qubo(qubo, n)
        classical_points.append(_selected_objective(c_bits, objectives))
    quantum = np.stack(quantum_points)
    classical = np.stack(classical_points)
    q_mask = pareto_front(quantum)
    c_mask = pareto_front(classical)
    pooled = np.concatenate([quantum, classical], axis=0)
    span = pooled.max(axis=0) - pooled.min(axis=0)
    reference = pooled.min(axis=0) - 0.1 * span - 1.0e-6
    ideal = pooled.max(axis=0)
    q_hv = normalized_hypervolume(quantum[q_mask], reference, ideal)
    c_hv = normalized_hypervolume(classical[c_mask], reference, ideal)
    return SweepResult(
        quantum_front=quantum[q_mask],
        classical_front=classical[c_mask],
        quantum_hypervolume=q_hv,
        classical_hypervolume=c_hv,
    )
