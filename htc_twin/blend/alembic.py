from __future__ import annotations

from typing import NamedTuple

import torch

from htc_twin.blend.qubo import Qubo


def all_bitstrings(n: int) -> torch.Tensor:
    space = 1 << n
    index = torch.arange(space)
    bits = torch.zeros(space, n)
    for qubit in range(n):
        bits[:, qubit] = ((index >> (n - 1 - qubit)) & 1).to(torch.float32)
    return bits


def cost_diagonal(qubo: Qubo, n: int) -> torch.Tensor:
    bits = all_bitstrings(n)
    quad = torch.einsum("zi,ij,zj->z", bits, qubo.matrix.to(bits.dtype), bits)
    return quad + qubo.constant


def apply_phase(state: torch.Tensor, diagonal: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    phase = torch.exp(-1j * gamma * diagonal)
    return state * phase


def apply_mixer(state: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    cos = torch.cos(beta).to(torch.complex64)
    sin = torch.sin(beta).to(torch.complex64)
    gate = torch.stack(
        [
            torch.stack([cos, -1j * sin]),
            torch.stack([-1j * sin, cos]),
        ]
    )
    tensor = state.reshape((2,) * n)
    for qubit in range(n):
        tensor = torch.tensordot(gate, tensor, dims=([1], [qubit]))
        tensor = torch.moveaxis(tensor, 0, qubit)
    return tensor.reshape(-1)


class QaoaResult(NamedTuple):
    probabilities: torch.Tensor
    expectation: float
    betas: torch.Tensor
    gammas: torch.Tensor


def qaoa_state(
    diagonal: torch.Tensor, n: int, betas: torch.Tensor, gammas: torch.Tensor
) -> torch.Tensor:
    space = 1 << n
    state = torch.full((space,), 1.0 / (space**0.5), dtype=torch.complex64)
    for layer in range(betas.shape[0]):
        state = apply_phase(state, diagonal, gammas[layer])
        state = apply_mixer(state, betas[layer], n)
    return state


def qaoa_expectation(
    diagonal: torch.Tensor, n: int, betas: torch.Tensor, gammas: torch.Tensor
) -> torch.Tensor:
    state = qaoa_state(diagonal, n, betas, gammas)
    probs = (state.conj() * state).real
    return (probs * diagonal).sum()


def _run_once(
    diagonal: torch.Tensor, n: int, depth: int, steps: int, lr: float, seed: int
) -> tuple[float, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    ramp = torch.linspace(0.1, 0.9, depth)
    betas = torch.nn.Parameter(
        torch.flip(ramp, dims=[0]) + 0.05 * torch.rand(depth, generator=generator)
    )
    gammas = torch.nn.Parameter(ramp + 0.05 * torch.rand(depth, generator=generator))
    optimiser = torch.optim.Adam([betas, gammas], lr=lr)
    for _ in range(steps):
        optimiser.zero_grad()
        loss = qaoa_expectation(diagonal, n, betas, gammas)
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        value = float(qaoa_expectation(diagonal, n, betas, gammas))
    return value, betas.detach(), gammas.detach()


def optimise_qaoa(
    qubo: Qubo,
    n: int,
    depth: int,
    steps: int,
    lr: float,
    seed: int,
    restarts: int = 4,
) -> QaoaResult:
    diagonal = cost_diagonal(qubo, n)
    best_value = float("inf")
    best_betas = torch.zeros(depth)
    best_gammas = torch.zeros(depth)
    for trial in range(max(1, restarts)):
        value, betas, gammas = _run_once(diagonal, n, depth, steps, lr, seed + 101 * trial)
        if value < best_value:
            best_value = value
            best_betas = betas
            best_gammas = gammas
    with torch.no_grad():
        state = qaoa_state(diagonal, n, best_betas, best_gammas)
        probs = (state.conj() * state).real
        expectation = float((probs * diagonal).sum())
    return QaoaResult(
        probabilities=probs,
        expectation=expectation,
        betas=best_betas,
        gammas=best_gammas,
    )


def most_probable_bitstring(probabilities: torch.Tensor, n: int) -> torch.Tensor:
    index = int(probabilities.argmax())
    bits = torch.zeros(n)
    for qubit in range(n):
        bits[qubit] = (index >> (n - 1 - qubit)) & 1
    return bits


def feasible_indices(n: int) -> list[int]:
    return [1 << (n - 1 - qubit) for qubit in range(n)]


def _index_to_bits(index: int, n: int) -> torch.Tensor:
    bits = torch.zeros(n)
    for qubit in range(n):
        bits[qubit] = (index >> (n - 1 - qubit)) & 1
    return bits


def best_feasible_bitstring(probabilities: torch.Tensor, n: int) -> torch.Tensor:
    indices = feasible_indices(n)
    masses = torch.tensor([float(probabilities[idx]) for idx in indices])
    return _index_to_bits(indices[int(masses.argmax())], n)


def decode_qaoa(
    probabilities: torch.Tensor, qubo: Qubo, n: int, shots: int, seed: int
) -> torch.Tensor:
    diagonal = cost_diagonal(qubo, n)
    feasible = set(feasible_indices(n))
    if shots <= 0:
        best = min(feasible_indices(n), key=lambda idx: float(diagonal[idx]))
        return _index_to_bits(best, n)
    generator = torch.Generator().manual_seed(seed)
    samples = torch.multinomial(probabilities, shots, replacement=True, generator=generator)
    best_index: int | None = None
    best_energy = float("inf")
    for sampled in samples.tolist():
        if sampled in feasible and float(diagonal[sampled]) < best_energy:
            best_energy = float(diagonal[sampled])
            best_index = sampled
    if best_index is None:
        best_index = min(feasible_indices(n), key=lambda idx: float(diagonal[idx]))
    return _index_to_bits(best_index, n)
