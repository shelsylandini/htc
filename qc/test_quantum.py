from __future__ import annotations

import numpy as np
import torch

from htc_twin.blend.alembic import (
    all_bitstrings,
    cost_diagonal,
    decode_qaoa,
    optimise_qaoa,
    qaoa_state,
)
from htc_twin.blend.marriage import pareto_front, weight_grid
from htc_twin.blend.qubo import build_qubo, objective_vector, one_hot_feasible, qubo_energy
from htc_twin.blend.vatting import brute_force_qubo, non_dominated_sort, nsga2


def _objectives(seed: int, n: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    raw = torch.rand(3, n, n, generator=generator)
    return 0.5 * (raw + raw.transpose(-1, -2))


def test_one_hot_feasibility() -> None:
    bits = torch.eye(4)
    assert torch.all(one_hot_feasible(bits))
    assert not bool(one_hot_feasible(torch.tensor([1.0, 1.0, 0.0, 0.0])))


def test_qubo_energy_matches_diagonal() -> None:
    objectives = _objectives(0, 4)
    weights = torch.tensor([0.5, 0.3, 0.2])
    qubo = build_qubo(weights, objectives, 4.0)
    diagonal = cost_diagonal(qubo, 4)
    bits = all_bitstrings(4)
    assert torch.allclose(qubo_energy(bits, qubo), diagonal, atol=1e-5)


def test_statevector_normalised() -> None:
    objectives = _objectives(1, 4)
    qubo = build_qubo(torch.tensor([0.4, 0.4, 0.2]), objectives, 4.0)
    diagonal = cost_diagonal(qubo, 4)
    state = qaoa_state(diagonal, 4, torch.tensor([0.3]), torch.tensor([0.5]))
    mass = float((state.conj() * state).real.sum())
    assert abs(mass - 1.0) < 1e-5


def test_qaoa_decode_matches_brute_force() -> None:
    agree = 0
    for seed in range(6):
        objectives = _objectives(seed, 5)
        qubo = build_qubo(torch.tensor([0.5, 0.3, 0.2]), objectives, 6.0)
        result = optimise_qaoa(qubo, 5, 3, 100, 0.08, seed, 4)
        decoded = decode_qaoa(result.probabilities, qubo, 5, 256, seed)
        brute = brute_force_qubo(qubo, 5)
        agree += int(torch.equal(decoded, brute))
    assert agree >= 5


def test_objective_vector_for_one_hot() -> None:
    objectives = _objectives(2, 4)
    bits = torch.tensor([0.0, 1.0, 0.0, 0.0])
    vector = objective_vector(bits, objectives)
    assert torch.allclose(vector, objectives[:, 1, 1], atol=1e-5)


def test_weight_grid_sums_to_one() -> None:
    grid = weight_grid(3, 4)
    for weights in grid:
        assert abs(sum(weights) - 1.0) < 1e-9


def test_pareto_front_selects_non_dominated() -> None:
    points = np.array([[1.0, 0.0], [0.0, 1.0], [0.4, 0.4], [0.5, 0.5]])
    mask = pareto_front(points)
    assert mask[0] and mask[1] and mask[3]
    assert not mask[2]


def test_non_dominated_sort_first_front() -> None:
    objectives = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    fronts = non_dominated_sort(objectives)
    assert 0 in fronts[0]


def test_nsga2_runs_and_returns_front() -> None:
    def evaluate(genome: np.ndarray) -> np.ndarray:
        ones = float(genome.sum())
        return np.array([ones, len(genome) - ones])

    front = nsga2(evaluate, 6, 12, 5, 0)
    assert front.shape[1] == 2
    assert front.shape[0] >= 1
