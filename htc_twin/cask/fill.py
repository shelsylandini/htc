from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch

from htc_twin.cask.optimisation import build_optimiser, build_scheduler
from htc_twin.cut.conservatism import cql_objective, td_target
from htc_twin.cut.sequence import dt_cross_entropy
from htc_twin.cut.setpoint import returns_to_go
from htc_twin.mash.cohort import LoggedDataset
from htc_twin.mash.wort import TransitionLoader, build_sequences, sample_sequences
from htc_twin.recipe import ActionGrid, ExperimentConfig
from htc_twin.still.column import HtcModel
from htc_twin.warehouse.logbook import get_logger
from htc_twin.warehouse.seeding import set_seed

LOGGER = get_logger("trainer")


class OfflineTrainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        total_steps: int,
        device: torch.device | None = None,
        dataset: LoggedDataset | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        set_seed(cfg.train.seed)
        self.generator = torch.Generator().manual_seed(cfg.train.seed)
        self.grid = ActionGrid(cfg.model.action)
        if dataset is None:
            raise ValueError("A caller-provided logged dataset is required for training.")
        self.dataset = dataset
        self.model = HtcModel(cfg.model, self.grid.size).to(self.device)
        self.loader = TransitionLoader(self.dataset)
        returns = returns_to_go(self.dataset.rewards, cfg.train.gamma, cfg.train.decay_window)
        self.sequences = build_sequences(self.dataset, returns)
        self.optimiser = build_optimiser(
            self.model.learnable_parameters(), cfg.train.lr, cfg.train.weight_decay
        )
        self.scheduler = build_scheduler(self.optimiser, cfg.train.warmup, max(1, total_steps))
        self.step = 0

    def train_step(self) -> dict[str, float]:
        self.model.train()
        batch = self.loader.sample(self.cfg.train.batch_size, self.generator)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        terminals = batch["terminals"].to(self.device)
        grades = batch["grades"].to(self.device)

        q_all = self.model.qnet(states)
        q_data = q_all.gather(1, actions.view(-1, 1).long()).squeeze(1)
        with torch.no_grad():
            next_value = self.model.target(next_states).max(dim=-1).values
        target = td_target(rewards, self.cfg.train.gamma, next_value, terminals)
        cql = cql_objective(q_all, q_data, target, self.cfg.train.cql_alpha)

        sequence = sample_sequences(self.sequences, self.cfg.train.batch_size, self.generator)
        logits = self.model.transformer(
            sequence["returns"].to(self.device),
            sequence["states"].to(self.device),
            sequence["actions"].to(self.device),
            sequence["timesteps"].to(self.device),
        )
        dt = dt_cross_entropy(
            logits, sequence["actions"].to(self.device), sequence["mask"].to(self.device)
        )

        probs = self.model.progression_probs(states)
        progression = -torch.log(probs.gather(1, grades.view(-1, 1).long()).squeeze(1)).mean()

        loss = cql.total + dt + progression
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.learnable_parameters(), self.cfg.train.grad_clip)
        self.optimiser.step()
        self.scheduler.step()
        self.model.sync_target(self.cfg.train.target_tau)
        self.step += 1
        return {
            "loss": float(loss.detach()),
            "cql": float(cql.total.detach()),
            "td": float(cql.temporal_difference.detach()),
            "conservative": float(cql.conservative.detach()),
            "dt": float(dt.detach()),
            "progression": float(progression.detach()),
        }

    def fit(self, steps: int, log_every: int = 10) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        for index in range(steps):
            metrics = self.train_step()
            history.append(metrics)
            if log_every > 0 and (index % log_every == 0 or index == steps - 1):
                LOGGER.info(
                    "step %d loss %.4f dt %.4f cql %.4f",
                    index,
                    metrics["loss"],
                    metrics["dt"],
                    metrics["cql"],
                )
        return history

    def config_payload(self) -> dict[str, Any]:
        return asdict(self.cfg)
