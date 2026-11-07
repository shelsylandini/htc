from __future__ import annotations

import torch

from htc_twin.mash.gauge import benefit_index
from htc_twin.recipe import ModelConfig, state_spec


class FeatureMap:
    def __init__(self, model: ModelConfig, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.model = model
        self.spec = state_spec(model)
        self.proj_act = torch.randn(4, model.dim_act, generator=generator)
        self.proj_clin = torch.randn(4, model.dim_clin, generator=generator)

    def encode(self, structural: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        cartilage = structural[:, 0]
        inflammation = structural[:, 1]
        gut = structural[:, 2]
        benefit = benefit_index(cartilage, inflammation) / 100.0
        core = torch.stack([cartilage, inflammation, gut, benefit], dim=-1)
        x_act = core @ self.proj_act
        x_clin = core @ self.proj_clin
        jsw = 2.5 * cartilage
        x_str = torch.stack([cartilage, inflammation, jsw], dim=-1)
        feature = torch.cat([x_act, x_clin, x_str, gut.unsqueeze(-1)], dim=-1)
        if noise is not None:
            feature = feature + noise
        return feature
