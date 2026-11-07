from __future__ import annotations

import torch

from htc_twin.kiln.anchors import anchor_book, anchor_grid, anchor_weighted_loss
from htc_twin.kiln.hazard import (
    convex_adaptation_value,
    empirical_hazard_curve,
    hazard_partition,
    paired_gain,
)
from htc_twin.kiln.ledger import discounted_returns, fold_by_patient
from htc_twin.kiln.prescription import (
    compare_fronts,
    enumerate_prescriptions,
    prescription_front,
    scalarized_choices,
    weight_grid,
)
from htc_twin.recipe import ActionConfig


def test_anchor_book_and_loss() -> None:
    book = anchor_book()
    grid = anchor_grid(book)
    assert torch.allclose(grid, torch.tensor([12.0, 9.6, 14.4, 16.0, 36.0]))
    predicted = torch.tensor([8.0, 12.5, 16.0, 40.0])
    observed = torch.tensor([9.0, 13.0, 13.5, 38.0])
    loss = anchor_weighted_loss(predicted, observed, book)
    assert torch.isfinite(loss)
    assert book.by_name("conservative_womac_total").contains(12.0)


def test_hazard_curve_checks_convex_gain() -> None:
    hazard = torch.linspace(0.0, 1.0, 30)
    baseline = 1.0 - 0.2 * hazard
    adaptive = baseline + convex_adaptation_value(hazard, base=0.5, alpha=0.8)
    curve = empirical_hazard_curve(hazard, baseline, adaptive, bins=6)
    assert curve.is_nonnegative()
    assert curve.is_increasing()
    assert curve.is_convex(tolerance=1.0e-4)
    partition = hazard_partition(hazard)
    assert partition.counts().sum().item() == 30
    gain, se = paired_gain(adaptive, baseline)
    assert gain > 0.0
    assert se >= 0.0


def test_fold_and_discounted_returns() -> None:
    fold = fold_by_patient(20, seed=2)
    assert fold.contains_once()
    assert fold.sizes() == (12, 3, 5)
    rewards = torch.ones(2, 4)
    returns = discounted_returns(rewards, gamma=0.5)
    assert torch.allclose(returns[0], torch.tensor([1.875, 1.75, 1.5, 1.0]))


def test_prescription_front_with_action_grid() -> None:
    from htc_twin.recipe import ActionGrid

    grid = ActionGrid(ActionConfig())
    scores = enumerate_prescriptions(grid)
    front = prescription_front(scores)
    weights = weight_grid(4)
    choices = scalarized_choices(scores, weights)
    assert len(scores) == 12
    assert front.hypervolume >= 0.0
    assert choices.numel() == weights.shape[0]
    comparison = compare_fronts(front, front)
    assert comparison["action_jaccard"] == 1.0
