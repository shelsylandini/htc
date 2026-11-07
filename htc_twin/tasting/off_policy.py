from __future__ import annotations

import torch

from htc_twin.mash.cohort import LoggedDataset
from htc_twin.still.lowwines import ConservativeQNet


def softmax_policy(
    qnet: ConservativeQNet, states: torch.Tensor, temperature: float
) -> torch.Tensor:
    with torch.no_grad():
        flat = qnet(states.reshape(-1, states.shape[-1]))
    probs = torch.softmax(flat / temperature, dim=-1)
    return probs.reshape(states.shape[0], states.shape[1], -1)


def discounted_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    horizon = rewards.shape[1]
    discount = gamma ** torch.arange(horizon, dtype=torch.float32)
    return (rewards * discount.unsqueeze(0)).sum(dim=1)


def per_step_ratio(
    behaviour: torch.Tensor, actions: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    chosen_target = target.gather(2, actions.unsqueeze(-1).long()).squeeze(-1)
    chosen_behaviour = behaviour[actions.long()]
    return chosen_target / chosen_behaviour.clamp_min(1.0e-8)


def trajectory_weights(ratios: torch.Tensor) -> tuple[torch.Tensor, float]:
    raw = ratios.clamp(0.0, 1.0e4).prod(dim=1)
    total = raw.sum().clamp_min(1.0e-8)
    normalized = raw / total
    n_eff = float((raw.sum() ** 2) / (raw.pow(2).sum().clamp_min(1.0e-12)))
    return normalized, n_eff


def weighted_is(rewards: torch.Tensor, ratios: torch.Tensor, gamma: float) -> tuple[float, float]:
    returns = discounted_return(rewards, gamma)
    weights, n_eff = trajectory_weights(ratios)
    value = float((weights * returns).sum())
    return value, n_eff


def state_value(probs: torch.Tensor, q_all: torch.Tensor) -> torch.Tensor:
    return (probs * q_all).sum(dim=-1)


def weighted_dr(
    rewards: torch.Tensor,
    ratios: torch.Tensor,
    target: torch.Tensor,
    q_all: torch.Tensor,
    actions: torch.Tensor,
    gamma: float,
) -> float:
    count, horizon = rewards.shape
    cumulative = ratios.clamp(0.0, 1.0e4).cumprod(dim=1)
    column_mass = cumulative.sum(dim=0, keepdim=True).clamp_min(1.0e-8)
    weight = cumulative / column_mass
    weight_prev = torch.full_like(weight, 1.0 / count)
    weight_prev[:, 1:] = weight[:, :-1]
    q_taken = q_all.gather(2, actions.unsqueeze(-1).long()).squeeze(-1)
    v_hat = state_value(target, q_all)
    discount = gamma ** torch.arange(horizon, dtype=torch.float32)
    term = weight * (rewards - q_taken) + weight_prev * v_hat
    return float((discount.unsqueeze(0) * term).sum().detach())


def fitted_q_eval(
    dataset: LoggedDataset,
    qnet: ConservativeQNet,
    temperature: float,
    gamma: float,
    iterations: int,
    lr: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    state_dim = dataset.state_dim
    fitted = ConservativeQNet(state_dim, qnet.n_actions, 64)
    optimiser = torch.optim.Adam(fitted.parameters(), lr=lr)
    states = dataset.states.reshape(-1, state_dim)
    next_states = dataset.next_states.reshape(-1, state_dim)
    actions = dataset.actions.reshape(-1).long()
    rewards = dataset.rewards.reshape(-1)
    terminals = dataset.terminals.reshape(-1)
    target_probs_next = softmax_policy(qnet, dataset.next_states, temperature).reshape(
        -1, qnet.n_actions
    )
    for _ in range(iterations):
        with torch.no_grad():
            next_q = fitted(next_states)
            bootstrap = (target_probs_next * next_q).sum(dim=-1)
            target = rewards + gamma * (1.0 - terminals) * bootstrap
        predicted = fitted(states).gather(1, actions.view(-1, 1)).squeeze(1)
        loss = torch.nn.functional.mse_loss(predicted, target)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        initial = dataset.states[:, 0, :]
        probs0 = softmax_policy(qnet, initial.unsqueeze(1), temperature).squeeze(1)
        q0 = fitted(initial)
        value = (probs0 * q0).sum(dim=-1).mean()
    return float(value)
