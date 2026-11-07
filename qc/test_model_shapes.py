from __future__ import annotations

import torch

from htc_twin.cut.conservatism import cql_objective, td_target
from htc_twin.still.lowwines import ConservativeQNet
from htc_twin.still.spirit import DecisionTransformer


def _make_dt() -> DecisionTransformer:
    return DecisionTransformer(
        state_dim=5, n_actions=4, width=16, layers=1, heads=2, max_timestep=8
    )


def test_decision_transformer_shape() -> None:
    model = _make_dt()
    returns = torch.zeros(3, 4)
    states = torch.randn(3, 4, 5)
    actions = torch.randint(0, 4, (3, 4))
    timesteps = torch.arange(4).unsqueeze(0).expand(3, 4)
    logits = model(returns, states, actions, timesteps)
    assert logits.shape == (3, 4, 4)


def test_decision_transformer_is_causal() -> None:
    model = _make_dt()
    model.eval()
    returns = torch.zeros(2, 4)
    states = torch.randn(2, 4, 5)
    timesteps = torch.arange(4).unsqueeze(0).expand(2, 4)
    actions_a = torch.zeros(2, 4, dtype=torch.long)
    actions_b = actions_a.clone()
    actions_b[:, -1] = 3
    with torch.no_grad():
        logits_a = model(returns, states, actions_a, timesteps)
        logits_b = model(returns, states, actions_b, timesteps)
    assert torch.allclose(logits_a[:, :-1], logits_b[:, :-1], atol=1e-5)


def test_decision_transformer_gradient_flow() -> None:
    model = _make_dt()
    returns = torch.zeros(2, 4)
    states = torch.randn(2, 4, 5)
    actions = torch.randint(0, 4, (2, 4))
    timesteps = torch.arange(4).unsqueeze(0).expand(2, 4)
    loss = model(returns, states, actions, timesteps).pow(2).mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None


def test_cql_conservative_sign() -> None:
    q_all = torch.tensor([[1.0, 2.0, 3.0]])
    q_data = torch.tensor([3.0])
    target = torch.tensor([3.0])
    terms = cql_objective(q_all, q_data, target, 1.0)
    assert float(terms.conservative) >= 0.0
    assert float(terms.temporal_difference) >= 0.0


def test_td_target_formula() -> None:
    reward = torch.tensor([1.0])
    nxt = torch.tensor([2.0])
    terminal = torch.tensor([0.0])
    assert abs(float(td_target(reward, 0.9, nxt, terminal)) - (1.0 + 0.9 * 2.0)) < 1e-5
    terminal = torch.tensor([1.0])
    assert abs(float(td_target(reward, 0.9, nxt, terminal)) - 1.0) < 1e-5


def test_qnet_greedy_and_gather() -> None:
    qnet = ConservativeQNet(5, 4, 8)
    state = torch.randn(6, 5)
    action = torch.randint(0, 4, (6,))
    assert qnet.gather(state, action).shape == (6,)
    assert qnet.greedy(state).shape == (6,)
