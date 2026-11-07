from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("width must divide into heads")
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        qkv = self.qkv(tokens)
        query, key, value = qkv.split(width, dim=2)
        query = query.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(length, length, device=tokens.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = torch.matmul(weights, value)
        merged = context.transpose(1, 2).reshape(batch, length, width)
        return self.proj(merged)


class Block(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class DecisionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        width: int,
        layers: int,
        heads: int,
        max_timestep: int,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.return_embed = nn.Linear(1, width)
        self.state_embed = nn.Linear(state_dim, width)
        self.action_embed = nn.Embedding(n_actions + 1, width)
        self.time_embed = nn.Embedding(max_timestep, width)
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.ModuleList([Block(width, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.action_head = nn.Linear(width, n_actions)
        self.width = width

    def forward(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        batch, length = actions.shape
        time = self.time_embed(timesteps.clamp_max(self.time_embed.num_embeddings - 1))
        ret_tokens = self.return_embed(returns.unsqueeze(-1)) + time
        state_tokens = self.state_embed(states) + time
        action_tokens = self.action_embed(actions.to(torch.long)) + time
        stacked = torch.stack([ret_tokens, state_tokens, action_tokens], dim=2)
        sequence = stacked.reshape(batch, length * 3, self.width)
        sequence = self.drop(sequence)
        for block in self.blocks:
            sequence = block(sequence)
        sequence = self.norm(sequence)
        state_positions = sequence[:, 1::3, :]
        return self.action_head(state_positions)

    def act(
        self,
        returns: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.forward(returns, states, actions, timesteps)
        return logits[:, -1, :]
