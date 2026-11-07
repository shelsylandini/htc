from __future__ import annotations

import torch


def mcid_reward(
    benefit: torch.Tensor,
    baseline: torch.Tensor,
    mcid: float,
    structural_change: torch.Tensor,
    switch_cost: torch.Tensor,
    lambda_s: float,
    lambda_a: float,
) -> torch.Tensor:
    setpoint = (benefit - baseline >= mcid).to(benefit.dtype)
    structural = torch.clamp_min(structural_change, 0.0)
    return setpoint - lambda_s * structural - lambda_a * switch_cost


def returns_to_go(rewards: torch.Tensor, gamma: float, decay_window: int) -> torch.Tensor:
    batch, horizon = rewards.shape
    output = torch.zeros_like(rewards)
    for start in range(horizon):
        acc = torch.zeros(batch, dtype=rewards.dtype, device=rewards.device)
        discount = 1.0
        stop = min(horizon, start + decay_window)
        for step in range(start, stop):
            acc = acc + discount * rewards[:, step]
            discount *= gamma
        output[:, start] = acc
    return output
