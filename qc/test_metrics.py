from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from htc_twin.nose.calibration import expected_calibration_error, ordinal_brier
from htc_twin.nose.discrimination import auroc
from htc_twin.nose.hypervolume import hypervolume_2d, normalized_hypervolume
from htc_twin.nose.responder import responder_gain


def test_auroc_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = (rng.random(200) < 1.0 / (1.0 + np.exp(-scores))).astype(np.int_)
    reference = roc_auc_score(labels, scores)
    assert abs(auroc(scores, labels) - reference) < 1e-9


def test_hypervolume_2d_known_value() -> None:
    points = np.array([[1.0, 0.5], [0.5, 1.0]])
    reference = np.array([0.0, 0.0])
    assert abs(hypervolume_2d(points, reference) - 0.75) < 1e-9


def test_normalized_hypervolume_in_unit_range() -> None:
    points = np.array([[0.2, 0.8], [0.8, 0.2], [0.5, 0.5]])
    reference = np.array([0.0, 0.0])
    ideal = np.array([1.0, 1.0])
    value = normalized_hypervolume(points, reference, ideal)
    assert 0.0 <= value <= 1.0
    assert value > 0.0


def test_responder_gain_sign() -> None:
    treated = np.full(100, 60.0)
    treated_base = np.full(100, 45.0)
    control = np.full(100, 50.0)
    control_base = np.full(100, 45.0)
    gain = responder_gain(treated, treated_base, control, control_base, 12.0)
    assert gain > 0.0


def test_calibration_bounds() -> None:
    probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 0])
    ece = expected_calibration_error(probs, labels, bins=5)
    assert 0.0 <= ece <= 1.0
    brier = ordinal_brier(probs, labels)
    assert brier >= 0.0
