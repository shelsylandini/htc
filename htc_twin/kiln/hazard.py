from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch


@dataclass(frozen=True)
class HazardPartition:
    labels: torch.Tensor
    cutpoints: tuple[float, float]
    names: tuple[str, str, str]

    def masks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (self.labels == 0, self.labels == 1, self.labels == 2)

    def counts(self) -> torch.Tensor:
        return torch.stack([(self.labels == index).sum() for index in range(3)])

    def name_for(self, index: int) -> str:
        return self.names[index]


@dataclass(frozen=True)
class HazardCurve:
    hazard: torch.Tensor
    baseline_value: torch.Tensor
    adaptive_value: torch.Tensor

    @property
    def delta(self) -> torch.Tensor:
        return self.adaptive_value - self.baseline_value

    def slopes(self) -> torch.Tensor:
        if self.hazard.numel() < 2:
            return torch.zeros(0, dtype=self.hazard.dtype, device=self.hazard.device)
        dx = self.hazard[1:] - self.hazard[:-1]
        dy = self.delta[1:] - self.delta[:-1]
        return dy / dx.clamp_min(torch.finfo(dx.dtype).eps)

    def curvature(self) -> torch.Tensor:
        slopes = self.slopes()
        if slopes.numel() < 2:
            return torch.zeros(0, dtype=self.hazard.dtype, device=self.hazard.device)
        left = self.hazard[:-2]
        right = self.hazard[2:]
        span = (right - left).clamp_min(torch.finfo(self.hazard.dtype).eps)
        return 2.0 * (slopes[1:] - slopes[:-1]) / span

    def is_nonnegative(self, tolerance: float = 1.0e-6) -> bool:
        return bool((self.delta >= -tolerance).all())

    def is_increasing(self, tolerance: float = 1.0e-6) -> bool:
        slopes = self.slopes()
        if slopes.numel() == 0:
            return True
        return bool((slopes >= -tolerance).all())

    def is_convex(self, tolerance: float = 1.0e-6) -> bool:
        curvature = self.curvature()
        if curvature.numel() == 0:
            return True
        return bool((curvature >= -tolerance).all())


def hazard_partition(
    hazard: torch.Tensor, lower: float | None = None, upper: float | None = None
) -> HazardPartition:
    flat = hazard.flatten().to(torch.float32)
    lower_value = float(torch.quantile(flat, 1.0 / 3.0).item()) if lower is None else lower
    upper_value = float(torch.quantile(flat, 2.0 / 3.0).item()) if upper is None else upper
    labels = torch.zeros_like(flat, dtype=torch.long)
    labels = torch.where(flat > lower_value, torch.ones_like(labels), labels)
    labels = torch.where(flat > upper_value, torch.full_like(labels, 2), labels)
    return HazardPartition(
        labels.reshape(hazard.shape),
        (lower_value, upper_value),
        ("stable", "intermediate", "incident"),
    )


def convex_adaptation_value(
    hazard: torch.Tensor, base: float, alpha: float, exponent: float = 2.0
) -> torch.Tensor:
    if base <= 0.0:
        raise ValueError("base")
    if alpha <= 0.0 or alpha > 1.0:
        raise ValueError("alpha")
    centered = hazard.to(torch.float32).clamp_min(0.0)
    return base * alpha * centered.pow(exponent)


def empirical_hazard_curve(
    hazard: torch.Tensor,
    baseline_value: torch.Tensor,
    adaptive_value: torch.Tensor,
    bins: int = 8,
) -> HazardCurve:
    if bins < 2:
        raise ValueError("bins")
    h = hazard.flatten().to(torch.float32)
    b = baseline_value.flatten().to(torch.float32)
    a = adaptive_value.flatten().to(torch.float32)
    order = torch.argsort(h)
    h = h[order]
    b = b[order]
    a = a[order]
    edges = torch.linspace(0, h.numel(), bins + 1, dtype=torch.long)
    hs = []
    bs = []
    av = []
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        start = int(left.item())
        stop = max(start + 1, int(right.item()))
        hs.append(h[start:stop].mean())
        bs.append(b[start:stop].mean())
        av.append(a[start:stop].mean())
    return HazardCurve(torch.stack(hs), torch.stack(bs), torch.stack(av))


def stratum_summary(values: torch.Tensor, partition: HazardPartition) -> dict[str, float]:
    out: dict[str, float] = {}
    for index, name in enumerate(partition.names):
        mask = partition.labels == index
        if bool(mask.any()):
            out[name] = float(values[mask].float().mean().item())
        else:
            out[name] = float("nan")
    return out


def hazard_from_state(
    state: torch.Tensor, cartilage_index: int, inflammation_index: int, gut_index: int
) -> torch.Tensor:
    cartilage = state[..., cartilage_index].to(torch.float32)
    inflammation = state[..., inflammation_index].to(torch.float32)
    gut = state[..., gut_index].to(torch.float32)
    cartilage_risk = (1.0 - cartilage).clamp(0.0, 1.0)
    inflammation_risk = inflammation.clamp_min(0.0) / (1.0 + inflammation.clamp_min(0.0))
    gut_risk = (1.0 - gut).clamp_min(0.0)
    return (0.55 * cartilage_risk + 0.35 * inflammation_risk + 0.10 * gut_risk).clamp(0.0, 1.0)


def odds_to_probability(odds: torch.Tensor) -> torch.Tensor:
    return odds / (1.0 + odds)


def probability_to_odds(probability: torch.Tensor) -> torch.Tensor:
    p = probability.clamp(1.0e-6, 1.0 - 1.0e-6)
    return p / (1.0 - p)


def standard_error(values: torch.Tensor) -> float:
    flat = values.flatten().to(torch.float32)
    if flat.numel() < 2:
        return 0.0
    return float((flat.std(unbiased=True) / sqrt(float(flat.numel()))).item())


def paired_gain(candidate: torch.Tensor, baseline: torch.Tensor) -> tuple[float, float]:
    diff = (candidate - baseline).flatten().to(torch.float32)
    return float(diff.mean().item()), standard_error(diff)


def monotone_projection(values: torch.Tensor) -> torch.Tensor:
    flat = values.flatten().to(torch.float32)
    out = flat.clone()
    for index in range(1, out.numel()):
        out[index] = torch.maximum(out[index], out[index - 1])
    return out.reshape(values.shape)


def convex_projection(values: torch.Tensor, passes: int = 8) -> torch.Tensor:
    if values.numel() < 3:
        return values.clone()
    out = monotone_projection(values)
    flat = out.flatten()
    for _ in range(passes):
        for index in range(1, flat.numel() - 1):
            floor = 0.5 * (flat[index - 1] + flat[index + 1])
            flat[index] = torch.minimum(flat[index], floor)
        flat = monotone_projection(flat)
    return flat.reshape(values.shape)
