from __future__ import annotations

import torch
from torch import nn

from htc_twin.recipe import TwinConfig


def anabolic_response(u: torch.Tensor, u_opt: float) -> torch.Tensor:
    ratio = u / u_opt
    return ratio * torch.exp(1.0 - ratio)


def antiinflammatory_response(u: torch.Tensor) -> torch.Tensor:
    return u / (1.0 + u)


def coupling_response(u: torch.Tensor) -> torch.Tensor:
    return u / (1.0 + u)


class GutJointODE(nn.Module):
    def __init__(self, cfg: TwinConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("k1", torch.tensor(cfg.k1))
        self.register_buffer("k2", torch.tensor(cfg.k2))
        self.register_buffer("k3", torch.tensor(cfg.k3))
        self.register_buffer("k4", torch.tensor(cfg.k4))
        self.register_buffer("k5", torch.tensor(cfg.k5))
        self.register_buffer("k6", torch.tensor(cfg.k6))
        self.register_buffer("k7", torch.tensor(cfg.k7))
        self.register_buffer("c_max", torch.tensor(cfg.c_max))
        self.register_buffer("s0", torch.tensor(cfg.s0))
        self.k1: torch.Tensor
        self.k2: torch.Tensor
        self.k3: torch.Tensor
        self.k4: torch.Tensor
        self.k5: torch.Tensor
        self.k6: torch.Tensor
        self.k7: torch.Tensor
        self.c_max: torch.Tensor
        self.s0: torch.Tensor

    def rhs(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        cartilage = state[..., 0]
        inflammation = state[..., 1]
        gut = state[..., 2]
        f_load = anabolic_response(u, self.cfg.u_opt)
        g_load = antiinflammatory_response(u)
        h_load = coupling_response(u)
        d_cart = -self.k1 * inflammation * cartilage + self.k2 * f_load * (self.c_max - cartilage)
        d_infl = self.k3 * gut - self.k4 * inflammation - self.k5 * g_load
        d_gut = self.k6 * (self.s0 - gut) - self.k7 * h_load
        return torch.stack([d_cart, d_infl, d_gut], dim=-1)

    def step(self, state: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        k1 = self.rhs(state, u)
        k2 = self.rhs(state + 0.5 * dt * k1, u)
        k3 = self.rhs(state + 0.5 * dt * k2, u)
        k4 = self.rhs(state + dt * k3, u)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def forward(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        current = state
        for _ in range(self.cfg.substeps):
            current = self.step(current, u, self.cfg.dt)
        return current

    def rollout(self, state: torch.Tensor, loads: torch.Tensor) -> torch.Tensor:
        horizon = loads.shape[1]
        trajectory = [state]
        current = state
        for epoch in range(horizon):
            current = self.forward(current, loads[:, epoch])
            trajectory.append(current)
        return torch.stack(trajectory, dim=1)
