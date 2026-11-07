from __future__ import annotations

import numpy as np
import numpy.typing as npt


def expected_calibration_error(
    probs: npt.NDArray[np.float64], labels: npt.NDArray[np.int_], bins: int = 10
) -> float:
    confidence = probs.max(axis=1)
    prediction = probs.argmax(axis=1)
    correct = (prediction == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    total = len(labels)
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lower) & (confidence <= upper)
        count = int(mask.sum())
        if count == 0:
            continue
        gap = abs(correct[mask].mean() - confidence[mask].mean())
        error += (count / total) * gap
    return float(error)


def ordinal_brier(probs: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    grades = probs.shape[1]
    cumulative_pred = np.cumsum(probs, axis=1)
    target = (labels[:, None] <= np.arange(grades)[None, :]).astype(np.float64)
    return float(((cumulative_pred - target) ** 2).sum(axis=1).mean())
