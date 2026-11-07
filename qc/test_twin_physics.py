from __future__ import annotations

import torch

from htc_twin.ferment.vat import GutJointODE, anabolic_response
from htc_twin.recipe import TwinConfig


def test_rhs_and_step_shape() -> None:
    twin = GutJointODE(TwinConfig())
    state = torch.tensor([[0.5, 0.3, 0.8], [0.7, 0.2, 0.9]])
    load = torch.tensor([0.85, 0.4])
    assert twin.rhs(state, load).shape == (2, 3)
    assert twin.forward(state, load).shape == (2, 3)


def test_anabolic_response_peaks_at_optimum() -> None:
    cfg = TwinConfig()
    grid = torch.linspace(0.05, 3.0, 200)
    values = anabolic_response(grid, cfg.u_opt)
    peak = grid[int(values.argmax())]
    assert abs(float(peak) - cfg.u_opt) < 0.05
    assert abs(float(anabolic_response(torch.tensor(cfg.u_opt), cfg.u_opt)) - 1.0) < 1e-5


def test_gut_axis_relaxes_to_baseline() -> None:
    cfg = TwinConfig()
    twin = GutJointODE(cfg)
    state = torch.tensor([[0.6, 0.3, 0.4]])
    zero = torch.zeros(1)
    for _ in range(40):
        state = twin.forward(state, zero)
    assert abs(float(state[0, 2]) - cfg.s0) < 0.05


def test_optimal_load_grows_cartilage() -> None:
    cfg = TwinConfig()
    twin = GutJointODE(cfg)
    start = torch.tensor([[0.4, 0.1, 0.9]])
    optimal = start.clone()
    idle = start.clone()
    for _ in range(20):
        optimal = twin.forward(optimal, torch.full((1,), cfg.u_opt))
        idle = twin.forward(idle, torch.zeros(1))
    assert float(optimal[0, 0]) > float(idle[0, 0])
