from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CohortShift:
    name: str
    mean_delta: torch.Tensor
    std_ratio: torch.Tensor
    auc_drop: float
    value_drop: float

    def magnitude(self) -> float:
        mean_score = float(torch.linalg.vector_norm(self.mean_delta).item())
        ratio_score = float(
            torch.linalg.vector_norm(torch.log(self.std_ratio.clamp_min(1.0e-6))).item()
        )
        return mean_score + ratio_score + abs(self.auc_drop) + abs(self.value_drop)


@dataclass(frozen=True)
class TransferAudit:
    reference: str
    shifts: tuple[CohortShift, ...]
    accepted_auc_drop: float
    accepted_value_drop: float

    def accepted(self) -> bool:
        for shift in self.shifts:
            if abs(shift.auc_drop) > self.accepted_auc_drop:
                return False
            if abs(shift.value_drop) > self.accepted_value_drop:
                return False
        return True

    def worst(self) -> CohortShift:
        if not self.shifts:
            raise ValueError("shifts")
        return max(self.shifts, key=lambda item: item.magnitude())


def cohort_shift(
    name: str,
    reference_features: torch.Tensor,
    target_features: torch.Tensor,
    reference_auc: float,
    target_auc: float,
    reference_value: float,
    target_value: float,
) -> CohortShift:
    ref = reference_features.to(torch.float32)
    target = target_features.to(torch.float32)
    if ref.ndim != 2 or target.ndim != 2:
        raise ValueError("features")
    mean_delta = target.mean(dim=0) - ref.mean(dim=0)
    ref_std = ref.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    target_std = target.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    return CohortShift(
        name=name,
        mean_delta=mean_delta,
        std_ratio=target_std / ref_std,
        auc_drop=target_auc - reference_auc,
        value_drop=target_value - reference_value,
    )


def transfer_audit(
    reference: str,
    shifts: tuple[CohortShift, ...],
    accepted_auc_drop: float = 0.088,
    accepted_value_drop: float = 0.039,
) -> TransferAudit:
    return TransferAudit(reference, shifts, accepted_auc_drop, accepted_value_drop)


def summarize_transfer(audit: TransferAudit) -> dict[str, float | str | bool]:
    if not audit.shifts:
        return {"reference": audit.reference, "accepted": True, "worst": "none", "magnitude": 0.0}
    worst = audit.worst()
    return {
        "reference": audit.reference,
        "accepted": audit.accepted(),
        "worst": worst.name,
        "magnitude": worst.magnitude(),
        "auc_drop": worst.auc_drop,
        "value_drop": worst.value_drop,
    }


def maximum_mean_discrepancy(
    left: torch.Tensor, right: torch.Tensor, bandwidth: float = 1.0
) -> torch.Tensor:
    if bandwidth <= 0.0:
        raise ValueError("bandwidth")
    x = left.to(torch.float32)
    y = right.to(torch.float32)
    xx = gaussian_kernel(x, x, bandwidth).mean()
    yy = gaussian_kernel(y, y, bandwidth).mean()
    xy = gaussian_kernel(x, y, bandwidth).mean()
    return xx + yy - 2.0 * xy


def gaussian_kernel(left: torch.Tensor, right: torch.Tensor, bandwidth: float) -> torch.Tensor:
    diff = left[:, None, :] - right[None, :, :]
    dist = diff.square().sum(dim=-1)
    return torch.exp(-dist / (2.0 * bandwidth * bandwidth))


def energy_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    x = left.to(torch.float32)
    y = right.to(torch.float32)
    xy = torch.cdist(x, y).mean()
    xx = torch.cdist(x, x).mean()
    yy = torch.cdist(y, y).mean()
    return 2.0 * xy - xx - yy


def calibration_slope(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    p = predicted.flatten().to(torch.float32)
    y = observed.flatten().to(torch.float32)
    centered_p = p - p.mean()
    centered_y = y - y.mean()
    denom = centered_p.square().sum().clamp_min(1.0e-6)
    return (centered_p * centered_y).sum() / denom


def calibration_intercept(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    slope = calibration_slope(predicted, observed)
    return observed.float().mean() - slope * predicted.float().mean()


def decile_table(predicted: torch.Tensor, observed: torch.Tensor, bins: int = 10) -> torch.Tensor:
    if bins < 2:
        raise ValueError("bins")
    p = predicted.flatten().to(torch.float32)
    y = observed.flatten().to(torch.float32)
    order = torch.argsort(p)
    p = p[order]
    y = y[order]
    edges = torch.linspace(0, p.numel(), bins + 1, dtype=torch.long)
    rows = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        start = int(left.item())
        stop = max(start + 1, int(right.item()))
        rows.append(torch.stack([p[start:stop].mean(), y[start:stop].mean()]))
    return torch.stack(rows, dim=0)


def brier_score(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    return (predicted.to(torch.float32) - observed.to(torch.float32)).square().mean()


def expected_calibration_error(
    predicted: torch.Tensor, observed: torch.Tensor, bins: int = 10
) -> torch.Tensor:
    table = decile_table(predicted, observed, bins)
    return (table[:, 0] - table[:, 1]).abs().mean()
