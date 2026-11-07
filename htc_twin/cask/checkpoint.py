from __future__ import annotations

from pathlib import Path
from typing import Any

from htc_twin.still.column import HtcModel
from htc_twin.warehouse.store import load_checkpoint, save_atomic


def save_model(
    path: str | Path,
    model: HtcModel,
    optimiser_state: dict[str, Any],
    seed: int,
    step: int,
    config: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimiser": optimiser_state,
        "seed": seed,
        "step": step,
        "config": config,
    }
    save_atomic(payload, path)


def restore_model(path: str | Path, model: HtcModel) -> dict[str, Any]:
    payload = load_checkpoint(path)
    model.load_state_dict(payload["model"])
    return payload
