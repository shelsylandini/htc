from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from simple_parsing import ArgumentParser

from htc_twin.bottling.loader import load_experiment
from htc_twin.cask.checkpoint import restore_model, save_model
from htc_twin.cask.controller import ClosedLoopController
from htc_twin.cask.fill import OfflineTrainer
from htc_twin.recipe import ActionGrid, ExperimentConfig, state_spec
from htc_twin.still.column import HtcModel
from htc_twin.tasting.report import (
    initial_structural,
    manuscript_reference,
    off_policy_table,
    progression_scores,
    quantum_summary,
)
from htc_twin.warehouse.logbook import get_logger

LOGGER = get_logger("stillhouse")


@dataclass
class Distil:
    config: str
    steps: int = 0
    out: str = "runs/closedloop_oa.pt"


@dataclass
class Taste:
    config: str
    ckpt: str | None = None
    steps: int = 0


@dataclass
class Pour:
    config: str
    ckpt: str | None = None
    patients: int = 64


@dataclass
class Blend:
    config: str


@dataclass
class Bottle:
    config: str
    ckpt: str | None = None
    out: str = "runs/value_network.onnx"


def _rebuild(cfg: ExperimentConfig, checkpoint: str | None) -> HtcModel:
    grid = ActionGrid(cfg.model.action)
    model = HtcModel(cfg.model, grid.size)
    if checkpoint is not None:
        restore_model(checkpoint, model)
    return model


def _distil(opts: Distil) -> None:
    cfg = load_experiment(opts.config)
    total = opts.steps if opts.steps > 0 else cfg.train.epochs
    trainer = OfflineTrainer(cfg, total)
    history = trainer.fit(total)
    save_model(
        opts.out,
        trainer.model,
        trainer.optimiser.state_dict(),
        cfg.train.seed,
        trainer.step,
        trainer.config_payload(),
    )
    LOGGER.info("new make spirit loss %.4f filled to %s", history[-1]["loss"], opts.out)


def _taste(opts: Taste) -> None:
    cfg = load_experiment(opts.config)
    if opts.ckpt is None:
        total = opts.steps if opts.steps > 0 else cfg.train.epochs
        trainer = OfflineTrainer(cfg, total)
        trainer.fit(total, log_every=0)
        model = trainer.model
        dataset = trainer.dataset
    else:
        model = _rebuild(cfg, opts.ckpt)
        dataset = OfflineTrainer(cfg, 1).dataset
    table = off_policy_table(model, dataset, cfg)
    progression = progression_scores(model, dataset)
    paper = manuscript_reference()
    LOGGER.info("WIS gain %+.2f%% paper %+.1f", table.wis_gain, paper["wis"])
    LOGGER.info("WDR gain %+.2f%% paper %+.1f", table.wdr_gain, paper["wdr"])
    LOGGER.info("FQE gain %+.2f%% paper %+.1f", table.fqe_gain, paper["fqe"])
    LOGGER.info("MCID responder %+.2fpp paper %+.1f", table.mcid_gain, paper["mcid"])
    LOGGER.info("stratum stable %+.3f incident %+.3f", table.stable_gain, table.incident_gain)
    LOGGER.info("progression AUROC %.3f paper %.3f", progression.overall, paper["progression_auc"])


def _pour(opts: Pour) -> None:
    cfg = load_experiment(opts.config)
    trainer = OfflineTrainer(cfg, 1)
    model = trainer.model if opts.ckpt is None else _rebuild(cfg, opts.ckpt)
    structural = initial_structural(trainer.dataset)[: opts.patients]
    controller = ClosedLoopController(model, cfg)
    rollout = controller.rollout(structural, trainer.dataset.horizon, cfg.train.target_return)
    LOGGER.info("mean closed-loop value %.4f", float(rollout["value"].mean()))
    LOGGER.info("leading prescriptions %s", rollout["actions"][:5, 0].tolist())


def _blend(opts: Blend) -> None:
    cfg = load_experiment(opts.config)
    summary = quantum_summary(cfg)
    paper = manuscript_reference()
    LOGGER.info(
        "QAOA hypervolume %.3f paper %.3f", summary["qaoa_hypervolume"], paper["qaoa_hypervolume"]
    )
    LOGGER.info(
        "classical hypervolume %.3f paper %.3f",
        summary["nsga_hypervolume"],
        paper["nsga_hypervolume"],
    )
    LOGGER.info("near-term advantage %+.4f", summary["advantage"])


def _bottle(opts: Bottle) -> None:
    cfg = load_experiment(opts.config)
    model = _rebuild(cfg, opts.ckpt)
    model.eval()
    grid = ActionGrid(cfg.model.action)
    example_input = torch.zeros(1, state_spec(cfg.model).dim)
    Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.qnet,
        (example_input,),
        opts.out,
        input_names=["state"],
        output_names=["q_values"],
        dynamo=False,
    )
    LOGGER.info("sealed value network to %s actions %d", opts.out, grid.size)


def main() -> None:
    parser = ArgumentParser()
    subs = parser.add_subparsers(dest="stage", required=True)
    subs.add_parser("distil").add_arguments(Distil, dest="opts")
    subs.add_parser("taste").add_arguments(Taste, dest="opts")
    subs.add_parser("pour").add_arguments(Pour, dest="opts")
    subs.add_parser("blend").add_arguments(Blend, dest="opts")
    subs.add_parser("bottle").add_arguments(Bottle, dest="opts")
    args = parser.parse_args()
    handlers: dict[str, Callable[[Any], None]] = {
        "distil": _distil,
        "taste": _taste,
        "pour": _pour,
        "blend": _blend,
        "bottle": _bottle,
    }
    handlers[args.stage](args.opts)


if __name__ == "__main__":
    main()
