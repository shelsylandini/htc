from __future__ import annotations

import torch

from htc_twin.recipe import ActionConfig, ActionGrid


def test_encode_decode_roundtrip() -> None:
    grid = ActionGrid(ActionConfig(modalities=3, intensities=2, frequencies=2))
    for index in range(grid.size):
        modality, intensity, frequency = grid.decode(index)
        assert grid.encode(modality, intensity, frequency) == index


def test_loads_monotone_in_dose() -> None:
    grid = ActionGrid(ActionConfig(modalities=2, intensities=2, frequencies=2))
    loads = grid.loads()
    assert loads.shape == (grid.size,)
    assert float(loads.min()) > 0.0


def test_switch_cost_zero_when_unchanged() -> None:
    grid = ActionGrid(ActionConfig(modalities=2, intensities=2, frequencies=2))
    action = torch.tensor([0, 3, 5])
    assert torch.allclose(grid.switch_cost(action, action), torch.zeros(3))
    changed = grid.switch_cost(torch.tensor([0]), torch.tensor([7]))
    assert float(changed) > 0.0
