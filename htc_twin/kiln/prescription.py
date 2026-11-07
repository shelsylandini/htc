from __future__ import annotations

from dataclasses import dataclass

import torch

from htc_twin.recipe import ActionGrid


@dataclass(frozen=True)
class PrescriptionScore:
    action: int
    modality: int
    intensity: int
    frequency: int
    pain: float
    structure: float
    adherence: float

    def objective_tensor(self) -> torch.Tensor:
        return torch.tensor([self.pain, self.structure, self.adherence], dtype=torch.float32)

    def scalar(self, weights: torch.Tensor) -> float:
        return float(torch.dot(self.objective_tensor(), weights.to(torch.float32)).item())


@dataclass(frozen=True)
class PrescriptionFront:
    scores: tuple[PrescriptionScore, ...]
    dominated: tuple[int, ...]
    hypervolume: float

    def actions(self) -> torch.Tensor:
        return torch.tensor([score.action for score in self.scores], dtype=torch.long)

    def objectives(self) -> torch.Tensor:
        if not self.scores:
            return torch.zeros(0, 3)
        return torch.stack([score.objective_tensor() for score in self.scores], dim=0)

    def best(self, weights: torch.Tensor) -> PrescriptionScore:
        if not self.scores:
            raise ValueError("scores")
        values = torch.tensor([score.scalar(weights) for score in self.scores], dtype=torch.float32)
        return self.scores[int(torch.argmax(values).item())]


def enumerate_prescriptions(grid: ActionGrid) -> tuple[PrescriptionScore, ...]:
    loads = grid.loads()
    out: list[PrescriptionScore] = []
    for action in range(grid.size):
        modality, intensity, frequency = grid.decode(action)
        load = float(loads[action].item())
        pain = 1.0 - abs(load - 0.85) / 1.7
        structure = 1.0 - max(load - 1.0, 0.0) / 1.7
        adherence = 1.0 - 0.12 * float(intensity) - 0.08 * float(frequency) - 0.05 * float(modality)
        out.append(
            PrescriptionScore(
                action,
                modality,
                intensity,
                frequency,
                max(0.0, min(1.0, pain)),
                max(0.0, min(1.0, structure)),
                max(0.0, min(1.0, adherence)),
            )
        )
    return tuple(out)


def dominates(left: PrescriptionScore, right: PrescriptionScore, tolerance: float = 1.0e-8) -> bool:
    a = left.objective_tensor()
    b = right.objective_tensor()
    no_worse = bool((a >= b - tolerance).all())
    better = bool((a > b + tolerance).any())
    return no_worse and better


def prescription_front(
    scores: tuple[PrescriptionScore, ...], reference: torch.Tensor | None = None
) -> PrescriptionFront:
    kept: list[PrescriptionScore] = []
    dominated: list[int] = []
    for score in scores:
        is_dominated = False
        for other in scores:
            if other.action != score.action and dominates(other, score):
                is_dominated = True
                break
        if is_dominated:
            dominated.append(score.action)
        else:
            kept.append(score)
    ref = torch.zeros(3) if reference is None else reference.to(torch.float32)
    hv = (
        hypervolume_3d(torch.stack([item.objective_tensor() for item in kept]), ref)
        if kept
        else 0.0
    )
    return PrescriptionFront(tuple(kept), tuple(dominated), hv)


def hypervolume_3d(points: torch.Tensor, reference: torch.Tensor) -> float:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points")
    clipped = torch.maximum(points.to(torch.float32), reference.to(torch.float32))
    order = torch.argsort(clipped[:, 0], descending=True)
    sorted_points = clipped[order]
    volume = 0.0
    prev_x = float(reference[0].item())
    for point in sorted_points.flip(0):
        x = float(point[0].item())
        width = max(0.0, x - prev_x)
        slice_points = sorted_points[sorted_points[:, 0] >= x][:, 1:]
        area = hypervolume_2d(slice_points, reference[1:])
        volume += width * area
        prev_x = x
    return volume


def hypervolume_2d(points: torch.Tensor, reference: torch.Tensor) -> float:
    if points.numel() == 0:
        return 0.0
    order = torch.argsort(points[:, 0])
    sorted_points = points[order]
    area = 0.0
    best_y = float(reference[1].item())
    prev_x = float(reference[0].item())
    for point in sorted_points:
        x = float(point[0].item())
        y = float(point[1].item())
        width = max(0.0, x - prev_x)
        area += width * max(0.0, best_y - float(reference[1].item()))
        best_y = max(best_y, y)
        prev_x = x
    tail_width = max(0.0, 1.0 - prev_x)
    area += tail_width * max(0.0, best_y - float(reference[1].item()))
    return area


def weight_grid(steps: int) -> torch.Tensor:
    if steps < 2:
        raise ValueError("steps")
    rows = []
    for i in range(steps):
        for j in range(steps - i):
            k = steps - 1 - i - j
            row = torch.tensor([i, j, k], dtype=torch.float32)
            rows.append(row / row.sum().clamp_min(1.0))
    return torch.stack(rows, dim=0)


def scalarized_choices(
    scores: tuple[PrescriptionScore, ...], weights: torch.Tensor
) -> torch.Tensor:
    choices = []
    objective = torch.stack([score.objective_tensor() for score in scores], dim=0)
    for weight in weights:
        values = objective @ weight.to(torch.float32)
        choices.append(int(torch.argmax(values).item()))
    return torch.tensor(choices, dtype=torch.long)


def compare_fronts(left: PrescriptionFront, right: PrescriptionFront) -> dict[str, float]:
    left_actions = set(int(item.action) for item in left.scores)
    right_actions = set(int(item.action) for item in right.scores)
    union = left_actions | right_actions
    overlap = left_actions & right_actions
    jaccard = 1.0 if not union else len(overlap) / len(union)
    return {
        "left_hypervolume": left.hypervolume,
        "right_hypervolume": right.hypervolume,
        "hypervolume_delta": left.hypervolume - right.hypervolume,
        "action_jaccard": float(jaccard),
    }


def action_penalty_matrix(scores: tuple[PrescriptionScore, ...], penalty: float) -> torch.Tensor:
    n = len(scores)
    matrix = torch.zeros(n, n, dtype=torch.float32)
    for i, left in enumerate(scores):
        for j, right in enumerate(scores):
            if i == j:
                continue
            modality_gap = abs(left.modality - right.modality)
            intensity_gap = abs(left.intensity - right.intensity)
            frequency_gap = abs(left.frequency - right.frequency)
            matrix[i, j] = penalty * float(modality_gap + intensity_gap + frequency_gap)
    return matrix


def prescription_table(scores: tuple[PrescriptionScore, ...]) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for score in scores:
        out.append(
            {
                "action": score.action,
                "modality": score.modality,
                "intensity": score.intensity,
                "frequency": score.frequency,
                "pain": score.pain,
                "structure": score.structure,
                "adherence": score.adherence,
            }
        )
    return out
