from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from htc_twin.blend.marriage import sweep
from htc_twin.cask.controller import ClosedLoopController
from htc_twin.ferment.vat import antiinflammatory_response, coupling_response
from htc_twin.mash.cohort import LoggedDataset
from htc_twin.nose.discrimination import auroc, stratified_auroc
from htc_twin.recipe import ActionGrid, ExperimentConfig
from htc_twin.still.column import HtcModel
from htc_twin.tasting.maturation import policy_value, value_of_adaptation
from htc_twin.tasting.off_policy import (
    discounted_return,
    fitted_q_eval,
    per_step_ratio,
    softmax_policy,
    weighted_dr,
    weighted_is,
)


@dataclass
class TableOne:
    behaviour_value: float
    wis: float
    wdr: float
    fqe: float
    wis_gain: float
    wdr_gain: float
    fqe_gain: float
    mcid_gain: float
    stable_gain: float
    incident_gain: float
    n_eff: float


@dataclass
class ProgressionScores:
    overall: float
    by_grade: dict[int, float]


def manuscript_reference() -> dict[str, float]:
    return {
        "wis": 11.7,
        "wdr": 10.9,
        "fqe": 12.3,
        "mcid": 9.6,
        "stable": 3.8,
        "incident": 17.5,
        "progression_auc": 0.81,
        "progression_kl2": 0.824,
        "progression_kl01": 0.781,
        "qaoa_hypervolume": 0.838,
        "nsga_hypervolume": 0.842,
    }


def _relative_gain(value: float, baseline: float) -> float:
    return 100.0 * (value - baseline) / (abs(baseline) + 1.0e-6)


def off_policy_table(
    model: HtcModel, dataset: LoggedDataset, cfg: ExperimentConfig, temperature: float = 0.5
) -> TableOne:
    gamma = cfg.train.gamma
    states = dataset.states
    actions = dataset.actions
    rewards = dataset.rewards
    target = softmax_policy(model.qnet, states, temperature)
    with torch.no_grad():
        q_all = model.qnet(states.reshape(-1, dataset.state_dim)).reshape(
            states.shape[0], states.shape[1], -1
        )
    ratios = per_step_ratio(dataset.behaviour, actions, target)
    behaviour_value = float(discounted_return(rewards, gamma).mean())
    wis, n_eff = weighted_is(rewards, ratios, gamma)
    wdr = weighted_dr(rewards, ratios, target, q_all, actions, gamma)
    fqe = fitted_q_eval(dataset, model.qnet, temperature, gamma, 60, 1.0e-3, cfg.evaluation.seed)

    controller = ClosedLoopController(model, cfg)
    structural0 = initial_structural(dataset)
    rollout = controller.rollout(structural0, dataset.horizon, cfg.train.target_return)
    learned_value = rollout["value"].numpy().astype(np.float64)
    behaviour_per_patient = policy_value(rewards, gamma).numpy().astype(np.float64)
    strata = dataset.strata.numpy().astype(np.int_)
    adaptation = value_of_adaptation(learned_value, behaviour_per_patient, strata)
    stable_gain = adaptation.get(0, 0.0)
    incident_gain = adaptation.get(1, 0.0)

    responder_learned = _responder_rate(rollout["rewards"])
    responder_behaviour = _responder_rate(rewards)
    mcid_gain = 100.0 * (responder_learned - responder_behaviour)
    return TableOne(
        behaviour_value=behaviour_value,
        wis=wis,
        wdr=wdr,
        fqe=fqe,
        wis_gain=_relative_gain(wis, behaviour_value),
        wdr_gain=_relative_gain(wdr, behaviour_value),
        fqe_gain=_relative_gain(fqe, behaviour_value),
        mcid_gain=mcid_gain,
        stable_gain=stable_gain,
        incident_gain=incident_gain,
        n_eff=n_eff,
    )


def initial_structural(dataset: LoggedDataset) -> torch.Tensor:
    cart0 = dataset.cartilage[:, 0]
    infl0 = ((80.0 * cart0 - dataset.baseline) / 20.0).clamp_min(0.0)
    gut0 = dataset.states[:, 0, dataset.state_dim - 1].clamp_min(0.05)
    return torch.stack([cart0, infl0, gut0], dim=-1)


def _responder_rate(rewards: torch.Tensor) -> float:
    reached = (rewards >= 1.0).any(dim=1).to(torch.float32)
    return float(reached.mean())


def progression_scores(model: HtcModel, dataset: LoggedDataset) -> ProgressionScores:
    with torch.no_grad():
        scores = (
            model.expected_grade(dataset.states.reshape(-1, dataset.state_dim))
            .numpy()
            .astype(np.float64)
        )
    grades = dataset.grades.reshape(-1).numpy().astype(np.int_)
    labels = (grades >= 2).astype(np.int_)
    overall = auroc(scores, labels)
    by_grade = stratified_auroc(scores, labels, np.clip(grades, 0, 4))
    return ProgressionScores(overall=overall, by_grade=by_grade)


def build_prescription_objectives(grid: ActionGrid, u_opt: float) -> torch.Tensor:
    loads = grid.loads()
    pain = antiinflammatory_response(loads)
    structure = loads / u_opt * torch.exp(1.0 - loads / u_opt)
    healing = coupling_response(loads) * torch.exp(-0.3 * torch.clamp_min(loads - u_opt, 0.0))
    diag = torch.stack([pain, structure, healing], dim=0)
    matrices = torch.stack([torch.diag(diag[k]) for k in range(diag.shape[0])], dim=0)
    return matrices


def quantum_summary(cfg: ExperimentConfig) -> dict[str, float]:
    grid = ActionGrid(cfg.model.action)
    objectives = build_prescription_objectives(grid, cfg.model.twin.u_opt)
    result = sweep(objectives, cfg.quantum)
    return {
        "qaoa_hypervolume": result.quantum_hypervolume,
        "nsga_hypervolume": result.classical_hypervolume,
        "advantage": result.advantage,
    }
