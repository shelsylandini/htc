from __future__ import annotations

import torch
from torch import nn

from htc_twin.ferment.hydrometer import LearnedResidual, OrdinalReadout
from htc_twin.recipe import ModelConfig, state_spec
from htc_twin.still.lowwines import ConservativeQNet
from htc_twin.still.spirit import DecisionTransformer


class HtcModel(nn.Module):
    def __init__(self, model: ModelConfig, n_actions: int, max_timestep: int = 64) -> None:
        super().__init__()
        spec = state_spec(model)
        dim = spec.dim
        self.qnet = ConservativeQNet(dim, n_actions, model.hidden)
        self.target = ConservativeQNet(dim, n_actions, model.hidden)
        self.target.load_state_dict(self.qnet.state_dict())
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.transformer = DecisionTransformer(
            dim, n_actions, model.hidden, model.dt_layers, model.dt_heads, max_timestep
        )
        self.severity_head = nn.Sequential(
            nn.Linear(dim, model.residual_hidden),
            nn.Tanh(),
            nn.Linear(model.residual_hidden, 1),
        )
        self.residual = LearnedResidual(dim, model.residual_hidden)
        self.ordinal = OrdinalReadout(model.ordinal_grades)

    def severity(self, state: torch.Tensor) -> torch.Tensor:
        return self.severity_head(state).squeeze(-1) + self.residual(state)

    def progression_probs(self, state: torch.Tensor) -> torch.Tensor:
        return self.ordinal.category_probs(self.severity(state))

    def expected_grade(self, state: torch.Tensor) -> torch.Tensor:
        return self.ordinal.expected_grade(self.severity(state))

    def learnable_parameters(self) -> list[nn.Parameter]:
        out: list[nn.Parameter] = []
        for module in (
            self.qnet,
            self.transformer,
            self.severity_head,
            self.residual,
            self.ordinal,
        ):
            out.extend(module.parameters())
        return out

    def sync_target(self, tau: float) -> None:
        with torch.no_grad():
            for online, target in zip(
                self.qnet.parameters(), self.target.parameters(), strict=True
            ):
                target.mul_(1.0 - tau).add_(tau * online)
