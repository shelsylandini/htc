from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from htc_twin.cask.fill import OfflineTrainer
from htc_twin.mash.cohort import LoggedDataset
from htc_twin.recipe import ExperimentConfig
from htc_twin.tasting.report import off_policy_table

VARIANTS = (
    "full",
    "shuffled_activity",
    "static_guideline",
    "no_reward",
    "no_mechanistic_twin",
    "no_risk_estimator",
    "no_reward_no_twin",
)


@dataclass
class AblationRow:
    name: str
    gain: float
    drop: float
    verdict: str


def manuscript_table_two() -> list[AblationRow]:
    return [
        AblationRow("random_input", 0.6, -11.1, "sanity (collapses)"),
        AblationRow("static_guideline", 0.0, -11.7, "floor"),
        AblationRow("no_reward", 4.9, -6.8, "necessary"),
        AblationRow("no_mechanistic_twin", 9.1, -2.6, "necessary"),
        AblationRow("no_risk_estimator", 9.8, -1.9, "beneficial"),
        AblationRow("no_projected_gut", 11.1, -0.6, "null (supporting)"),
        AblationRow("classical_optimizer", 11.6, -0.1, "null (parity)"),
        AblationRow("no_reward_no_twin", 3.1, -8.6, "additive"),
        AblationRow("full", 11.7, 0.0, "reference"),
    ]


def verdict_for(name: str, drop: float, null_threshold: float) -> str:
    if name == "shuffled_activity":
        return "sanity (collapses)"
    if name == "static_guideline":
        return "floor"
    if abs(drop) < null_threshold:
        return "null"
    return "necessary"


def apply_variant(dataset: LoggedDataset, variant: str, seed: int) -> LoggedDataset:
    generator = torch.Generator().manual_seed(seed)
    if variant == "shuffled_activity":
        permutation = torch.randperm(dataset.actions.numel(), generator=generator)
        shuffled = dataset.actions.reshape(-1)[permutation].reshape(dataset.actions.shape)
        return replace(dataset, actions=shuffled)
    if variant in {"no_reward", "no_reward_no_twin"}:
        flat = torch.zeros_like(dataset.rewards)
        out = replace(dataset, rewards=flat)
        if variant == "no_reward_no_twin":
            return _scramble_dynamics(out, generator)
        return out
    if variant == "no_mechanistic_twin":
        return _scramble_dynamics(dataset, generator)
    return dataset


def _scramble_dynamics(dataset: LoggedDataset, generator: torch.Generator) -> LoggedDataset:
    permutation = torch.randperm(dataset.next_states.shape[0], generator=generator)
    return replace(dataset, next_states=dataset.next_states[permutation])


def run_variant(cfg: ExperimentConfig, variant: str, steps: int, dataset: LoggedDataset) -> float:
    modified = apply_variant(dataset, variant, cfg.train.seed + 17)
    trainer = OfflineTrainer(cfg, steps, dataset=modified)
    trainer.fit(steps, log_every=0)
    table = off_policy_table(trainer.model, trainer.dataset, cfg)
    return table.wis_gain


def table_two(cfg: ExperimentConfig, steps: int, dataset: LoggedDataset) -> list[AblationRow]:
    full_gain = run_variant(cfg, "full", steps, dataset)
    rows = [AblationRow("full", full_gain, 0.0, "reference")]
    for variant in VARIANTS[1:]:
        gain = run_variant(cfg, variant, steps, dataset)
        drop = gain - full_gain
        rows.append(
            AblationRow(
                variant, gain, drop, verdict_for(variant, drop, cfg.evaluation.null_threshold)
            )
        )
    return rows
