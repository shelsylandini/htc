from __future__ import annotations

import numpy as np
import numpy.typing as npt


def is_dominated(
    point: npt.NDArray[np.float64], front: npt.NDArray[np.float64]
) -> npt.NDArray[np.bool_]:
    return np.all(front >= point, axis=1)


def hypervolume_2d(points: npt.NDArray[np.float64], reference: npt.NDArray[np.float64]) -> float:
    if points.size == 0:
        return 0.0
    keep = points[np.all(points >= reference, axis=1)]
    if keep.size == 0:
        return 0.0
    order = np.argsort(-keep[:, 0], kind="mergesort")
    ordered = keep[order]
    area = 0.0
    prev_y = float(reference[1])
    for row in ordered:
        x_value = float(row[0])
        y_value = float(row[1])
        if y_value <= prev_y:
            continue
        area += (x_value - float(reference[0])) * (y_value - prev_y)
        prev_y = y_value
    return area


def hypervolume(
    points: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
    samples: int = 40000,
    seed: int = 0,
) -> float:
    if points.size == 0:
        return 0.0
    dim = points.shape[1]
    if dim == 2:
        upper = np.maximum(points.max(axis=0), reference + 1e-9)
        box = float(np.prod(upper - reference))
        if box <= 0.0:
            return 0.0
        return hypervolume_2d(points, reference)
    upper = np.maximum(points.max(axis=0), reference + 1e-9)
    rng = np.random.default_rng(seed)
    grid = rng.uniform(reference, upper, size=(samples, dim))
    dominated = np.zeros(samples, dtype=bool)
    for point in points:
        dominated |= np.all(grid <= point, axis=1)
    box = float(np.prod(upper - reference))
    return box * float(dominated.mean())


def normalized_hypervolume(
    points: npt.NDArray[np.float64],
    reference: npt.NDArray[np.float64],
    ideal: npt.NDArray[np.float64],
) -> float:
    span = ideal - reference
    span = np.where(span <= 0.0, 1.0, span)
    scaled = (points - reference) / span
    unit_reference = np.zeros_like(reference)
    raw = hypervolume(scaled, unit_reference)
    return float(np.clip(raw, 0.0, 1.0))
