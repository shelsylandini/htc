from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import torch

from htc_twin.mash.cohort import LoggedDataset


def policy_value(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    horizon = rewards.shape[1]
    discount = gamma ** torch.arange(horizon, dtype=torch.float32)
    return (rewards * discount.unsqueeze(0)).sum(dim=1)


def risk_score(dataset: LoggedDataset) -> npt.NDArray[np.float64]:
    base = dataset.baseline.numpy().astype(np.float64)
    span = base.max() - base.min()
    if span <= 0.0:
        return np.zeros_like(base)
    return (base.max() - base) / span


def risk_bins(score: npt.NDArray[np.float64], bins: int) -> npt.NDArray[np.int_]:
    edges = np.quantile(score, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.digitize(score, edges).astype(np.int_)


def value_of_adaptation(
    target_value: npt.NDArray[np.float64],
    behaviour_value: npt.NDArray[np.float64],
    strata: npt.NDArray[np.int_],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for level in np.unique(strata):
        mask = strata == level
        out[int(level)] = float((target_value[mask] - behaviour_value[mask]).mean())
    return out


class ConvexityCertificate(NamedTuple):
    non_negative: bool
    increasing: bool
    convex: bool

    @property
    def holds(self) -> bool:
        return self.non_negative and self.increasing and self.convex


def convexity_certificate(
    delta: npt.NDArray[np.float64], tolerance: float = 1.0e-6
) -> ConvexityCertificate:
    non_negative = bool(np.all(delta >= -tolerance))
    first = np.diff(delta)
    increasing = bool(np.all(first >= -tolerance))
    second = np.diff(first)
    convex = bool(np.all(second >= -tolerance))
    return ConvexityCertificate(non_negative=non_negative, increasing=increasing, convex=convex)


def adaptation_value(
    b0: float, alpha: float, g_values: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return b0 * alpha * g_values
