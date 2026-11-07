from __future__ import annotations

import numpy as np
import numpy.typing as npt


def responder_rate(
    benefit: npt.NDArray[np.float64], baseline: npt.NDArray[np.float64], mcid: float
) -> float:
    reached = (benefit - baseline) >= mcid
    return float(reached.mean())


def responder_gain(
    treated_benefit: npt.NDArray[np.float64],
    treated_baseline: npt.NDArray[np.float64],
    control_benefit: npt.NDArray[np.float64],
    control_baseline: npt.NDArray[np.float64],
    mcid: float,
) -> float:
    treated = responder_rate(treated_benefit, treated_baseline, mcid)
    control = responder_rate(control_benefit, control_baseline, mcid)
    return 100.0 * (treated - control)
