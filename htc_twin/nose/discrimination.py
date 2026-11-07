from __future__ import annotations

import numpy as np
import numpy.typing as npt


def auroc(scores: npt.NDArray[np.float64], labels: npt.NDArray[np.int_]) -> float:
    order = np.argsort(scores, kind="mergesort")
    ranked = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    index = 0
    rank_value = 1.0
    while index < len(ranked):
        stop = index
        while stop < len(ranked) and ranked[stop] == ranked[index]:
            stop += 1
        average = (rank_value + (rank_value + (stop - index) - 1.0)) / 2.0
        ranks[order[index:stop]] = average
        rank_value += stop - index
        index = stop
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    rank_sum = float(ranks[positives].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def stratified_auroc(
    scores: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int_],
    strata: npt.NDArray[np.int_],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for value in np.unique(strata):
        mask = strata == value
        out[int(value)] = auroc(scores[mask], labels[mask])
    return out
