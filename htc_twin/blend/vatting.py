from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import torch

from htc_twin.blend.alembic import all_bitstrings
from htc_twin.blend.qubo import Qubo, qubo_energy

Evaluator = Callable[[npt.NDArray[np.int_]], npt.NDArray[np.float64]]


def brute_force_qubo(qubo: Qubo, n: int) -> torch.Tensor:
    bits = all_bitstrings(n)
    energies = qubo_energy(bits, qubo)
    return bits[int(energies.argmin())]


def anneal_qubo(qubo: Qubo, n: int, steps: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    state = (rng.random(n) < 0.5).astype(np.float64)
    current = torch.from_numpy(state).to(torch.float32)
    energy = float(qubo_energy(current, qubo))
    best = current.clone()
    best_energy = energy
    for step in range(steps):
        temperature = max(1e-3, 1.0 - step / steps)
        flip = int(rng.integers(0, n))
        candidate = current.clone()
        candidate[flip] = 1.0 - candidate[flip]
        candidate_energy = float(qubo_energy(candidate, qubo))
        delta = candidate_energy - energy
        if delta < 0 or rng.random() < np.exp(-delta / temperature):
            current = candidate
            energy = candidate_energy
            if energy < best_energy:
                best = current.clone()
                best_energy = energy
    return best


def dominates(left: npt.NDArray[np.float64], right: npt.NDArray[np.float64]) -> bool:
    return bool(np.all(left <= right) and np.any(left < right))


def non_dominated_sort(objectives: npt.NDArray[np.float64]) -> list[list[int]]:
    count = objectives.shape[0]
    dominated: list[list[int]] = [[] for _ in range(count)]
    strikes = np.zeros(count, dtype=int)
    fronts: list[list[int]] = [[]]
    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dominated[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                strikes[i] += 1
        if strikes[i] == 0:
            fronts[0].append(i)
    cursor = 0
    while fronts[cursor]:
        nxt: list[int] = []
        for i in fronts[cursor]:
            for j in dominated[i]:
                strikes[j] -= 1
                if strikes[j] == 0:
                    nxt.append(j)
        cursor += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(
    objectives: npt.NDArray[np.float64], front: list[int]
) -> npt.NDArray[np.float64]:
    size = len(front)
    distance = np.zeros(size, dtype=np.float64)
    if size <= 2:
        return np.full(size, np.inf)
    block = objectives[front]
    for axis in range(block.shape[1]):
        order = np.argsort(block[:, axis])
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = block[order[-1], axis] - block[order[0], axis]
        if span <= 0:
            continue
        for rank in range(1, size - 1):
            distance[order[rank]] += (
                block[order[rank + 1], axis] - block[order[rank - 1], axis]
            ) / span
    return distance


def nsga2(
    evaluate: Evaluator,
    n_bits: int,
    pop_size: int,
    generations: int,
    seed: int,
) -> npt.NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    population = (rng.random((pop_size, n_bits)) < 0.5).astype(np.int_)
    objectives = np.stack([evaluate(genome) for genome in population])
    for _ in range(generations):
        offspring = _breed(population, rng)
        child_obj = np.stack([evaluate(genome) for genome in offspring])
        merged = np.concatenate([population, offspring], axis=0)
        merged_obj = np.concatenate([objectives, child_obj], axis=0)
        population, objectives = _select(merged, merged_obj, pop_size)
    fronts = non_dominated_sort(objectives)
    return objectives[fronts[0]]


def _breed(population: npt.NDArray[np.int_], rng: np.random.Generator) -> npt.NDArray[np.int_]:
    size, bits = population.shape
    children = np.empty_like(population)
    for index in range(size):
        a = population[int(rng.integers(0, size))]
        b = population[int(rng.integers(0, size))]
        crossover = rng.random(bits) < 0.5
        child = np.where(crossover, a, b)
        mutation = rng.random(bits) < (1.0 / bits)
        child = np.where(mutation, 1 - child, child)
        children[index] = child
    return children


def _select(
    population: npt.NDArray[np.int_],
    objectives: npt.NDArray[np.float64],
    pop_size: int,
) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.float64]]:
    fronts = non_dominated_sort(objectives)
    chosen: list[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= pop_size:
            chosen.extend(front)
            continue
        distance = crowding_distance(objectives, front)
        order = np.argsort(-distance)
        for rank in order:
            if len(chosen) >= pop_size:
                break
            chosen.append(front[int(rank)])
        break
    selection = np.array(chosen[:pop_size], dtype=int)
    return population[selection], objectives[selection]
