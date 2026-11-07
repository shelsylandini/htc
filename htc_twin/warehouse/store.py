from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import torch


def save_atomic(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, target)


def load_pickle(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        obj = pickle.load(handle)
    if not isinstance(obj, dict):
        raise ValueError("expected a mapping payload")
    return obj


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint payload must be a mapping")
    return payload
