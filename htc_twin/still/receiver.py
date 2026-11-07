from __future__ import annotations

import torch
from torch import nn

from htc_twin.still.lowwines import ConservativeQNet
from htc_twin.still.spirit import DecisionTransformer


class HtcPolicy(nn.Module):
    def __init__(
        self, qnet: ConservativeQNet, transformer: DecisionTransformer, top_k: int = 3
    ) -> None:
        super().__init__()
        self.qnet = qnet
        self.transformer = transformer
        self.top_k = top_k

    def deterministic_action(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.transformer.act(returns, states, actions, timesteps)
        width = min(self.top_k, logits.shape[-1])
        candidates = logits.topk(width, dim=-1).indices
        current = states[:, -1, :]
        values = self.qnet(current)
        gathered = values.gather(1, candidates)
        choice = gathered.argmax(dim=-1, keepdim=True)
        return candidates.gather(1, choice).squeeze(-1)

    def value_action(self, state: torch.Tensor) -> torch.Tensor:
        return self.qnet.greedy(state)
