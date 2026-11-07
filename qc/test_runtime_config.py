from __future__ import annotations

from pathlib import Path

import pytest

from htc_twin.bottling.loader import CaskError, load_experiment

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "recipes" / "batch"


def test_load_main_batch() -> None:
    cfg = load_experiment(BATCH / "main.toml")
    assert cfg.train.gamma == 0.95
    assert cfg.model.ordinal_grades == 5
    assert cfg.model.twin.k1 == 0.85


def test_extends_merges_parent() -> None:
    cfg = load_experiment(BATCH / "ablation_qaoa_depth1.toml")
    assert cfg.quantum.depth == 1
    assert cfg.train.gamma == 0.95
    assert cfg.name == "ablation_qaoa_depth1"


def test_unknown_key_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "bad"\n[train]\nnot_a_real_key = 1\n')
    with pytest.raises(CaskError):
        load_experiment(bad)
