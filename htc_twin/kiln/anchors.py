from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import torch


@dataclass(frozen=True)
class AnchorBand:
    name: str
    center: float
    lower: float
    upper: float
    source: str

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def threshold_tensor(self, device: torch.device | None = None) -> torch.Tensor:
        return torch.tensor(
            [self.lower, self.center, self.upper], dtype=torch.float32, device=device
        )

    def widen(self, ratio: float) -> AnchorBand:
        width = self.upper - self.lower
        extra = width * ratio * 0.5
        return AnchorBand(
            self.name, self.center, self.lower - extra, self.upper + extra, self.source
        )

    def clamp(self, floor: float = 0.0) -> AnchorBand:
        return AnchorBand(
            self.name, self.center, max(floor, self.lower), max(floor, self.upper), self.source
        )


@dataclass(frozen=True)
class McidAnchorBook:
    conservative: AnchorBand
    sensitivity_low: AnchorBand
    sensitivity_high: AnchorBand
    surgical_function: AnchorBand
    surgical_total: AnchorBand

    def bands(self) -> tuple[AnchorBand, ...]:
        return (
            self.conservative,
            self.sensitivity_low,
            self.sensitivity_high,
            self.surgical_function,
            self.surgical_total,
        )

    def by_name(self, name: str) -> AnchorBand:
        for band in self.bands():
            if band.name == name:
                return band
        raise KeyError(name)

    def responder_matrix(self, benefits: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        delta = benefits - baseline
        columns = []
        for band in self.bands():
            threshold = torch.tensor(band.center, dtype=delta.dtype, device=delta.device)
            columns.append((delta >= threshold).to(delta.dtype))
        return torch.stack(columns, dim=-1)

    def conservative_responder(
        self, benefits: torch.Tensor, baseline: torch.Tensor
    ) -> torch.Tensor:
        delta = benefits - baseline
        return (delta >= self.conservative.center).to(delta.dtype)

    def sensitivity_stack(self, benefits: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        delta = benefits - baseline
        low = (delta >= self.sensitivity_low.center).to(delta.dtype)
        mid = (delta >= self.conservative.center).to(delta.dtype)
        high = (delta >= self.sensitivity_high.center).to(delta.dtype)
        return torch.stack([low, mid, high], dim=-1)


def anchor_book() -> McidAnchorBook:
    conservative = AnchorBand(
        "conservative_womac_total",
        12.0,
        9.6,
        14.4,
        "Methods VI off-policy evaluation",
    )
    sensitivity_low = AnchorBand(
        "minus_twenty_percent",
        9.6,
        9.6,
        9.6,
        "Methods VI off-policy evaluation",
    )
    sensitivity_high = AnchorBand(
        "plus_twenty_percent",
        14.4,
        14.4,
        14.4,
        "Methods VI off-policy evaluation",
    )
    surgical_function = AnchorBand(
        "surgical_function",
        16.0,
        16.0,
        16.0,
        "Methods VI off-policy evaluation",
    )
    surgical_total = AnchorBand(
        "surgical_total",
        36.0,
        36.0,
        36.0,
        "Methods VI off-policy evaluation",
    )
    return McidAnchorBook(
        conservative,
        sensitivity_low,
        sensitivity_high,
        surgical_function,
        surgical_total,
    )


def anchor_grid(book: McidAnchorBook | None = None) -> torch.Tensor:
    active = anchor_book() if book is None else book
    return torch.tensor([band.center for band in active.bands()], dtype=torch.float32)


def responder_gain(
    candidate_benefits: torch.Tensor,
    baseline_benefits: torch.Tensor,
    initial_benefits: torch.Tensor,
    band: AnchorBand | None = None,
) -> torch.Tensor:
    active = anchor_book().conservative if band is None else band
    candidate_delta = candidate_benefits - initial_benefits
    baseline_delta = baseline_benefits - initial_benefits
    threshold = torch.tensor(
        active.center, dtype=candidate_delta.dtype, device=candidate_delta.device
    )
    candidate = (candidate_delta >= threshold).to(candidate_delta.dtype)
    baseline = (baseline_delta >= threshold).to(candidate_delta.dtype)
    return candidate.mean() - baseline.mean()


def stable_anchor_mean(values: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    mask = (values >= lower) & (values <= upper)
    if bool(mask.any()):
        return values[mask].mean()
    return values.mean()


def anchor_weighted_loss(
    predicted_delta: torch.Tensor,
    observed_delta: torch.Tensor,
    book: McidAnchorBook | None = None,
) -> torch.Tensor:
    active = anchor_book() if book is None else book
    loss = torch.zeros((), dtype=predicted_delta.dtype, device=predicted_delta.device)
    for band in active.bands():
        width = max(band.upper - band.lower, 1.0)
        threshold = torch.tensor(
            band.center, dtype=predicted_delta.dtype, device=predicted_delta.device
        )
        pred_margin = torch.sigmoid((predicted_delta - threshold) / width)
        obs_margin = (observed_delta >= threshold).to(predicted_delta.dtype)
        loss = loss + torch.nn.functional.binary_cross_entropy(pred_margin, obs_margin)
    return loss / len(active.bands())


def parse_anchor_rows(rows: list[dict[str, str]]) -> McidAnchorBook:
    parsed: dict[str, AnchorBand] = {}
    for row in rows:
        name = row["name"]
        center = float(row["center"])
        lower = float(row.get("lower", center))
        upper = float(row.get("upper", center))
        source = row.get("source", "manual")
        if not all(isfinite(value) for value in (center, lower, upper)):
            raise ValueError(name)
        parsed[name] = AnchorBand(name, center, lower, upper, source)
    required = [
        "conservative_womac_total",
        "minus_twenty_percent",
        "plus_twenty_percent",
        "surgical_function",
        "surgical_total",
    ]
    missing = [name for name in required if name not in parsed]
    if missing:
        raise KeyError(",".join(missing))
    return McidAnchorBook(
        parsed["conservative_womac_total"],
        parsed["minus_twenty_percent"],
        parsed["plus_twenty_percent"],
        parsed["surgical_function"],
        parsed["surgical_total"],
    )


def anchor_report(book: McidAnchorBook | None = None) -> list[dict[str, float | str]]:
    active = anchor_book() if book is None else book
    out: list[dict[str, float | str]] = []
    for band in active.bands():
        out.append(
            {
                "name": band.name,
                "center": band.center,
                "lower": band.lower,
                "upper": band.upper,
                "source": band.source,
            }
        )
    return out
