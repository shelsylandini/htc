from __future__ import annotations

from htc_twin.recipe import (
    ActionConfig,
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    QuantumConfig,
    TrainConfig,
)


def tiny_config(patients: int = 32, horizon: int = 4) -> ExperimentConfig:
    model = ModelConfig(
        hidden=16,
        dt_layers=1,
        dt_heads=2,
        context=horizon,
        residual_hidden=8,
        action=ActionConfig(modalities=2, intensities=2, frequencies=2),
    )
    data = DataConfig(patients=patients, epochs_per_patient=horizon, seed=0)
    train = TrainConfig(batch_size=16, epochs=2, lr=3.0e-3, warmup=0, decay_window=3, seed=0)
    quantum = QuantumConfig(depth=1, grid=3, opt_steps=8, restarts=1, shots=32, seed=0)
    evaluation = EvalConfig(bootstrap=16, seeds=2, seed=0)
    return ExperimentConfig("test", model, data, train, quantum, evaluation)
