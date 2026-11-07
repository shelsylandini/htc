from __future__ import annotations

import numpy as np

from htc_twin.tasting.bootstrap import clustered_bootstrap, effective_n
from htc_twin.tasting.maturation import adaptation_value, convexity_certificate, value_of_adaptation


def test_convexity_certificate_accepts_convex_increasing() -> None:
    delta = np.array([0.0, 0.5, 1.5, 3.0])
    certificate = convexity_certificate(delta)
    assert certificate.holds


def test_convexity_certificate_rejects_concave() -> None:
    delta = np.array([0.0, 2.0, 2.5, 2.6])
    certificate = convexity_certificate(delta)
    assert not certificate.convex


def test_proposition_one_curve_is_increasing() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    curve = adaptation_value(0.8, 0.3, rho**2)
    assert np.all(np.diff(curve) >= 0.0)
    assert np.all(curve >= 0.0)


def test_value_of_adaptation_by_stratum() -> None:
    target = np.array([1.0, 2.0, 3.0, 4.0])
    behaviour = np.array([0.5, 1.0, 1.5, 2.0])
    strata = np.array([0, 0, 1, 1])
    out = value_of_adaptation(target, behaviour, strata)
    assert out[0] > 0.0 and out[1] > out[0]


def test_effective_n_equal_weights() -> None:
    weights = np.ones(8)
    assert abs(effective_n(weights) - 8.0) < 1e-9


def test_clustered_bootstrap_brackets_mean() -> None:
    values = np.linspace(0.0, 1.0, 50)
    clusters = np.repeat(np.arange(10), 5)
    interval = clustered_bootstrap(values, clusters, 200, 0.05, 0)
    assert interval.lower <= interval.mean <= interval.upper
