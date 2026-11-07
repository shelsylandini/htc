from __future__ import annotations

from dataclasses import dataclass

import torch

from htc_twin.mash.cohort import LoggedDataset


@dataclass(frozen=True)
class EpochLedger:
    patient: int
    epoch: int
    action: int
    reward: float
    terminal: bool
    stratum: int
    baseline: float
    cartilage: float
    next_cartilage: float
    grade: int

    def structural_delta(self) -> float:
        return self.cartilage - self.next_cartilage

    def response_delta(self) -> float:
        return self.reward + self.baseline


@dataclass(frozen=True)
class PatientLedger:
    patient: int
    epochs: tuple[EpochLedger, ...]

    @property
    def horizon(self) -> int:
        return len(self.epochs)

    @property
    def stratum(self) -> int:
        if not self.epochs:
            return 0
        return self.epochs[0].stratum

    def actions(self) -> torch.Tensor:
        return torch.tensor([epoch.action for epoch in self.epochs], dtype=torch.long)

    def rewards(self) -> torch.Tensor:
        return torch.tensor([epoch.reward for epoch in self.epochs], dtype=torch.float32)

    def cartilage(self) -> torch.Tensor:
        return torch.tensor([epoch.cartilage for epoch in self.epochs], dtype=torch.float32)

    def terminal_epoch(self) -> int:
        for epoch in self.epochs:
            if epoch.terminal:
                return epoch.epoch
        return self.horizon - 1

    def cumulative_reward(self, gamma: float = 1.0) -> float:
        total = 0.0
        scale = 1.0
        for epoch in self.epochs:
            total += scale * epoch.reward
            scale *= gamma
        return total

    def switches(self) -> int:
        if len(self.epochs) < 2:
            return 0
        total = 0
        previous = self.epochs[0].action
        for epoch in self.epochs[1:]:
            if epoch.action != previous:
                total += 1
            previous = epoch.action
        return total

    def final_grade(self) -> int:
        if not self.epochs:
            return 0
        return self.epochs[-1].grade

    def first_grade(self) -> int:
        if not self.epochs:
            return 0
        return self.epochs[0].grade

    def progressed(self) -> bool:
        return self.final_grade() > self.first_grade()


@dataclass(frozen=True)
class CaseFold:
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor

    def all_ids(self) -> torch.Tensor:
        return torch.cat([self.train, self.validation, self.test], dim=0)

    def sizes(self) -> tuple[int, int, int]:
        return int(self.train.numel()), int(self.validation.numel()), int(self.test.numel())

    def contains_once(self) -> bool:
        ids = self.all_ids()
        return int(torch.unique(ids).numel()) == int(ids.numel())


def ledger_from_tensors(
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    strata: torch.Tensor,
    baseline: torch.Tensor,
    cartilage: torch.Tensor,
    next_cartilage: torch.Tensor,
    grades: torch.Tensor,
) -> tuple[EpochLedger, ...]:
    if actions.ndim != 2:
        raise ValueError("actions")
    n_patients, horizon = actions.shape
    out: list[EpochLedger] = []
    for patient in range(n_patients):
        for epoch in range(horizon):
            out.append(
                EpochLedger(
                    patient=patient,
                    epoch=epoch,
                    action=int(actions[patient, epoch].item()),
                    reward=float(rewards[patient, epoch].item()),
                    terminal=bool(terminals[patient, epoch].item() > 0.5),
                    stratum=int(strata[patient].item()),
                    baseline=float(baseline[patient].item()),
                    cartilage=float(cartilage[patient, epoch].item()),
                    next_cartilage=float(next_cartilage[patient, epoch].item()),
                    grade=int(grades[patient, epoch].item()),
                )
            )
    return tuple(out)


def flatten_logged_dataset(dataset: LoggedDataset) -> tuple[EpochLedger, ...]:
    return ledger_from_tensors(
        dataset.actions,
        dataset.rewards,
        dataset.terminals,
        dataset.strata,
        dataset.baseline,
        dataset.cartilage,
        dataset.next_cartilage,
        dataset.grades,
    )


def patient_ledgers(rows: tuple[EpochLedger, ...]) -> tuple[PatientLedger, ...]:
    grouped: dict[int, list[EpochLedger]] = {}
    for row in rows:
        grouped.setdefault(row.patient, []).append(row)
    out: list[PatientLedger] = []
    for patient in sorted(grouped):
        epochs = tuple(sorted(grouped[patient], key=lambda item: item.epoch))
        out.append(PatientLedger(patient, epochs))
    return tuple(out)


def fold_by_patient(
    patients: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.15,
    seed: int = 0,
) -> CaseFold:
    if train_fraction <= 0.0 or validation_fraction <= 0.0:
        raise ValueError("fractions")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("fractions")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(patients, generator=generator)
    train_end = int(round(patients * train_fraction))
    val_end = train_end + int(round(patients * validation_fraction))
    return CaseFold(order[:train_end], order[train_end:val_end], order[val_end:])


def discounted_returns(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    if rewards.ndim != 2:
        raise ValueError("rewards")
    out = torch.zeros_like(rewards)
    running = torch.zeros(rewards.shape[0], dtype=rewards.dtype, device=rewards.device)
    for index in range(rewards.shape[1] - 1, -1, -1):
        running = rewards[:, index] + gamma * running
        out[:, index] = running
    return out


def censor_after(terminals: torch.Tensor) -> torch.Tensor:
    if terminals.ndim != 2:
        raise ValueError("terminals")
    alive = torch.ones_like(terminals)
    closed = torch.zeros(terminals.shape[0], dtype=torch.bool, device=terminals.device)
    for epoch in range(terminals.shape[1]):
        alive[:, epoch] = (~closed).to(terminals.dtype)
        closed = closed | (terminals[:, epoch] > 0.5)
    return alive


def stitch_rewards(
    primary: torch.Tensor, fallback: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if primary.shape != fallback.shape or primary.shape != mask.shape:
        raise ValueError("shape")
    return torch.where(mask > 0.5, primary, fallback)


def patient_summary(rows: tuple[PatientLedger, ...], gamma: float = 1.0) -> dict[str, float]:
    if not rows:
        return {"patients": 0.0, "reward": 0.0, "switches": 0.0, "progression": 0.0}
    rewards = torch.tensor([row.cumulative_reward(gamma) for row in rows], dtype=torch.float32)
    switches = torch.tensor([row.switches() for row in rows], dtype=torch.float32)
    progression = torch.tensor(
        [1.0 if row.progressed() else 0.0 for row in rows], dtype=torch.float32
    )
    return {
        "patients": float(len(rows)),
        "reward": float(rewards.mean().item()),
        "switches": float(switches.mean().item()),
        "progression": float(progression.mean().item()),
    }


def action_histogram(rows: tuple[EpochLedger, ...], action_count: int) -> torch.Tensor:
    hist = torch.zeros(action_count, dtype=torch.float32)
    for row in rows:
        if 0 <= row.action < action_count:
            hist[row.action] += 1.0
    denom = hist.sum().clamp_min(1.0)
    return hist / denom


def transition_table(rows: tuple[PatientLedger, ...], action_count: int) -> torch.Tensor:
    table = torch.zeros(action_count, action_count, dtype=torch.float32)
    for patient in rows:
        actions = patient.actions()
        if actions.numel() < 2:
            continue
        for left, right in zip(actions[:-1], actions[1:], strict=True):
            table[int(left.item()), int(right.item())] += 1.0
    row_sum = table.sum(dim=1, keepdim=True).clamp_min(1.0)
    return table / row_sum


def take_fold(dataset: LoggedDataset, patient_ids: torch.Tensor) -> LoggedDataset:
    ids = patient_ids.to(torch.long)
    return LoggedDataset(
        states=dataset.states[ids],
        actions=dataset.actions[ids],
        rewards=dataset.rewards[ids],
        next_states=dataset.next_states[ids],
        terminals=dataset.terminals[ids],
        timesteps=dataset.timesteps[ids],
        strata=dataset.strata[ids],
        baseline=dataset.baseline[ids],
        cartilage=dataset.cartilage[ids],
        next_cartilage=dataset.next_cartilage[ids],
        grades=dataset.grades[ids],
        behaviour=dataset.behaviour,
    )
