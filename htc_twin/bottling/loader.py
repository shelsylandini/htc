from __future__ import annotations

import dataclasses
import tomllib
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

from htc_twin.recipe import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    ModelConfig,
    QuantumConfig,
    TrainConfig,
)

T = TypeVar("T")

_SECTIONS: dict[str, type] = {
    "model": ModelConfig,
    "data": DataConfig,
    "train": TrainConfig,
    "quantum": QuantumConfig,
    "evaluation": EvalConfig,
}


class CaskError(ValueError):
    pass


def _build(cls: type[T], payload: dict[str, Any], where: str) -> T:
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for key, value in payload.items():
        if key not in hints:
            raise CaskError(f"{where}: unexpected key '{key}'")
        annotation = hints[key]
        if dataclasses.is_dataclass(annotation) and isinstance(value, dict):
            kwargs[key] = _build(annotation, value, f"{where}.{key}")
        else:
            kwargs[key] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in over.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _resolve(path: Path, seen: set[str]) -> dict[str, Any]:
    raw = _read(path)
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise CaskError(f"{path}: 'extends' must be a cask name")
    if parent in seen:
        raise CaskError(f"{path}: circular extends via '{parent}'")
    seen.add(parent)
    base = _resolve(path.parent / f"{parent}.toml", seen)
    return _deep_merge(base, raw)


def load_experiment(path: str | Path) -> ExperimentConfig:
    target = Path(path)
    raw = _resolve(target, set())
    name = str(raw.pop("name", target.stem))
    built: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        chunk = raw.pop(key, None)
        if chunk is None:
            built[key] = cls()
        elif isinstance(chunk, dict):
            built[key] = _build(cls, chunk, key)
        else:
            raise CaskError(f"section '{key}' must be a table")
    if raw:
        raise CaskError(f"{target}: unexpected sections {sorted(raw)}")
    return ExperimentConfig(
        name=name,
        model=built["model"],
        data=built["data"],
        train=built["train"],
        quantum=built["quantum"],
        evaluation=built["evaluation"],
    )
