from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt


class Interval(NamedTuple):
    mean: float
    lower: float
    upper: float


def effective_n(weights: npt.NDArray[np.float64]) -> float:
    total = float(weights.sum())
    squared = float((weights**2).sum())
    if squared <= 0.0:
        return 0.0
    return (total**2) / squared


def clustered_bootstrap(
    values: npt.NDArray[np.float64],
    clusters: npt.NDArray[np.int_] | None,
    resamples: int,
    alpha: float,
    seed: int,
) -> Interval:
    rng = np.random.default_rng(seed)
    if clusters is None:
        clusters = np.arange(len(values))
    unique = np.unique(clusters)
    grouped = [values[clusters == key] for key in unique]
    means = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        pick = rng.integers(0, len(grouped), size=len(grouped))
        pooled = np.concatenate([grouped[choice] for choice in pick])
        means[index] = pooled.mean()
    lower = float(np.percentile(means, 100.0 * alpha / 2.0))
    upper = float(np.percentile(means, 100.0 * (1.0 - alpha / 2.0)))
    return Interval(mean=float(values.mean()), lower=lower, upper=upper)
