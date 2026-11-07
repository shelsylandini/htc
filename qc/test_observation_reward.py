from __future__ import annotations

import torch

from htc_twin.cut.setpoint import mcid_reward, returns_to_go
from htc_twin.ferment.hydrometer import OrdinalReadout, severity_latent


def test_thresholds_are_ordered() -> None:
    readout = OrdinalReadout(5)
    tau = readout.thresholds()
    assert torch.all(tau[1:] >= tau[:-1])


def test_category_probs_normalised() -> None:
    readout = OrdinalReadout(5)
    latent = torch.linspace(-2.0, 2.0, 16)
    probs = readout.category_probs(latent)
    assert probs.shape == (16, 5)
    assert torch.allclose(probs.sum(dim=1), torch.ones(16), atol=1e-5)


def test_expected_grade_monotone_in_severity() -> None:
    readout = OrdinalReadout(5)
    with torch.no_grad():
        readout.raw_weight.fill_(1.0)
        low = readout.expected_grade(torch.tensor([-1.0]))
        high = readout.expected_grade(torch.tensor([2.0]))
    assert float(high) > float(low)


def test_severity_latent_increases_with_damage() -> None:
    cart = torch.tensor([0.9, 0.2])
    residual = torch.zeros(2)
    latent = severity_latent(cart, 1.0, residual)
    assert float(latent[1]) > float(latent[0])


def test_mcid_reward_indicator() -> None:
    benefit = torch.tensor([60.0, 50.0])
    baseline = torch.tensor([45.0, 45.0])
    reward = mcid_reward(benefit, baseline, 12.0, torch.zeros(2), torch.zeros(2), 0.5, 0.1)
    assert float(reward[0]) == 1.0
    assert float(reward[1]) == 0.0


def test_returns_to_go_decay_window() -> None:
    rewards = torch.ones(1, 5)
    full = returns_to_go(rewards, 1.0, 5)
    windowed = returns_to_go(rewards, 1.0, 2)
    assert float(full[0, 0]) == 5.0
    assert float(windowed[0, 0]) == 2.0
