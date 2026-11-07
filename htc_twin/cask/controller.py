from __future__ import annotations

from typing import TypedDict

import torch

from htc_twin.cut.setpoint import mcid_reward
from htc_twin.ferment.vat import GutJointODE
from htc_twin.mash.gauge import benefit_index
from htc_twin.mash.grist import FeatureMap
from htc_twin.recipe import ActionGrid, ExperimentConfig
from htc_twin.still.column import HtcModel
from htc_twin.still.receiver import HtcPolicy


class Rollout(TypedDict):
    rewards: torch.Tensor
    value: torch.Tensor
    actions: torch.Tensor
    final_structural: torch.Tensor


class ClosedLoopController:
    def __init__(self, model: HtcModel, cfg: ExperimentConfig, top_k: int = 3) -> None:
        self.model = model
        self.cfg = cfg
        self.policy = HtcPolicy(model.qnet, model.transformer, top_k)
        self.twin = GutJointODE(cfg.model.twin)
        self.grid = ActionGrid(cfg.model.action)
        self.features = FeatureMap(cfg.model, cfg.data.seed)

    @torch.no_grad()
    def rollout(self, structural: torch.Tensor, horizon: int, target_return: float) -> Rollout:
        self.model.eval()
        batch = structural.shape[0]
        loads = self.grid.loads()
        baseline = benefit_index(structural[:, 0], structural[:, 1])
        previous = torch.zeros(batch, dtype=torch.long)
        return_ctx: list[torch.Tensor] = []
        state_ctx: list[torch.Tensor] = []
        action_ctx: list[torch.Tensor] = []
        time_ctx: list[torch.Tensor] = []
        rewards = torch.zeros(batch, horizon)
        actions = torch.zeros(batch, horizon, dtype=torch.long)
        for epoch in range(horizon):
            state = self.features.encode(structural)
            state_ctx.append(state)
            return_ctx.append(torch.full((batch,), target_return))
            time_ctx.append(torch.full((batch,), epoch, dtype=torch.long))
            pending = action_ctx + [torch.zeros(batch, dtype=torch.long)]
            chosen = self.policy.deterministic_action(
                torch.stack(return_ctx, dim=1),
                torch.stack(state_ctx, dim=1),
                torch.stack(pending, dim=1),
                torch.stack(time_ctx, dim=1),
            )
            action_ctx.append(chosen)
            load = loads[chosen]
            advanced = self.twin.forward(structural, load)
            advanced[:, 0] = advanced[:, 0].clamp(0.0, self.cfg.model.twin.c_max)
            advanced[:, 1] = advanced[:, 1].clamp_min(0.0)
            next_benefit = benefit_index(advanced[:, 0], advanced[:, 1])
            structural_change = (structural[:, 0] - advanced[:, 0]).clamp_min(0.0)
            switch = self.grid.switch_cost(previous, chosen)
            rewards[:, epoch] = mcid_reward(
                next_benefit,
                baseline,
                self.cfg.train.mcid,
                structural_change,
                switch,
                self.cfg.train.lambda_s,
                self.cfg.train.lambda_a,
            )
            actions[:, epoch] = chosen
            previous = chosen
            structural = advanced
        discount = self.cfg.train.gamma ** torch.arange(horizon, dtype=torch.float32)
        value = (rewards * discount.unsqueeze(0)).sum(dim=1)
        return Rollout(rewards=rewards, value=value, actions=actions, final_structural=structural)
