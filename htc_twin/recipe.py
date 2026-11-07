from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class ActionConfig:
    modalities: int = 3
    intensities: int = 2
    frequencies: int = 2
    load_gain: float = 1.6


@dataclass(frozen=True)
class TwinConfig:
    k1: float = 0.85
    k2: float = 0.65
    k3: float = 0.55
    k4: float = 0.70
    k5: float = 0.30
    k6: float = 0.50
    k7: float = 0.22
    c_max: float = 1.0
    s0: float = 1.0
    u_opt: float = 0.85
    dt: float = 0.25
    substeps: int = 4


@dataclass(frozen=True)
class ModelConfig:
    dim_act: int = 8
    dim_clin: int = 6
    dim_str: int = 3
    hidden: int = 128
    dt_layers: int = 2
    dt_heads: int = 4
    context: int = 20
    ordinal_grades: int = 5
    residual_hidden: int = 32
    twin: TwinConfig = field(default_factory=TwinConfig)
    action: ActionConfig = field(default_factory=ActionConfig)


@dataclass(frozen=True)
class DataConfig:
    patients: int = 4000
    epochs_per_patient: int = 12
    incident_fraction: float = 0.4
    womac_baseline: float = 48.0
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    grad_accum: int = 4
    epochs: int = 50
    lr: float = 3.0e-4
    warmup: int = 200
    weight_decay: float = 1.0e-4
    gamma: float = 0.95
    cql_alpha: float = 1.0
    target_tau: float = 5.0e-3
    mcid: float = 12.0
    lambda_s: float = 0.5
    lambda_a: float = 0.1
    decay_window: int = 6
    target_return: float = 1.0
    grad_clip: float = 1.0
    amp: bool = False
    seed: int = 0


@dataclass(frozen=True)
class QuantumConfig:
    depth: int = 3
    grid: int = 11
    penalty: float = 4.0
    opt_steps: int = 80
    opt_lr: float = 0.1
    restarts: int = 4
    shots: int = 256
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    bootstrap: int = 1000
    null_threshold: float = 1.0
    auc_tolerance: float = 1.5
    seeds: int = 10
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    data: DataConfig
    train: TrainConfig
    quantum: QuantumConfig
    evaluation: EvalConfig


@dataclass(frozen=True)
class StateSpec:
    dim_act: int
    dim_clin: int
    dim_str: int

    @property
    def dim(self) -> int:
        return self.dim_act + self.dim_clin + self.dim_str + 1

    @property
    def gut_index(self) -> int:
        return self.dim - 1

    @property
    def cartilage_index(self) -> int:
        return self.dim_act + self.dim_clin

    def structural_slice(self) -> slice:
        start = self.dim_act + self.dim_clin
        return slice(start, start + self.dim_str)


def state_spec(cfg: ModelConfig) -> StateSpec:
    return StateSpec(cfg.dim_act, cfg.dim_clin, cfg.dim_str)


class ActionGrid:
    def __init__(self, cfg: ActionConfig) -> None:
        self.cfg = cfg
        self.size = cfg.modalities * cfg.intensities * cfg.frequencies

    def decode(self, index: int) -> tuple[int, int, int]:
        freq = index % self.cfg.frequencies
        inten = (index // self.cfg.frequencies) % self.cfg.intensities
        modality = index // (self.cfg.frequencies * self.cfg.intensities)
        return modality, inten, freq

    def encode(self, modality: int, intensity: int, frequency: int) -> int:
        return (
            modality * self.cfg.intensities * self.cfg.frequencies
            + intensity * self.cfg.frequencies
            + frequency
        )

    def loads(self) -> torch.Tensor:
        out = torch.zeros(self.size)
        for index in range(self.size):
            _, inten, freq = self.decode(index)
            intensity_norm = (inten + 1.0) / self.cfg.intensities
            frequency_norm = (freq + 1.0) / self.cfg.frequencies
            out[index] = self.cfg.load_gain * intensity_norm * frequency_norm
        return out

    def switch_cost(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        prev = previous.to(torch.long)
        curr = current.to(torch.long)
        prev_mod = torch.div(
            prev, self.cfg.intensities * self.cfg.frequencies, rounding_mode="floor"
        )
        curr_mod = torch.div(
            curr, self.cfg.intensities * self.cfg.frequencies, rounding_mode="floor"
        )
        modality_change = (prev_mod != curr_mod).to(torch.float32)
        dose_change = (prev != curr).to(torch.float32)
        return 0.5 * modality_change + 0.5 * dose_change
