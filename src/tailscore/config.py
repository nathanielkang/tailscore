"""YAML-backed experiment configuration (§37 parity with PROPOSAL)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CorruptionConfig:
    train: str = "mdm"
    eval: str = "clean_cat"


@dataclass
class TCSEConfig:
    T: int = 50
    k_neighbors: int = 32
    lambda_weight: float = 1.0
    clip_upper: float = 5.0
    score_epochs: int = 2
    reg_epochs: int = 50
    # Two-stage protocol: fraction of training data used as score partition S.
    # R = 1 - ratio is used exclusively for regression.
    # Set to 1.0 for single-stage (backward compat); 0.5 recommended for R6.
    score_partition_ratio: float = 1.0
    # Weight blend: 0 = pure local kNN rank, 1 = pure global rank.
    global_rank_weight: float = 0.6
    # Power on blended rank before scaling (>=1 sharpens tail upweighting).
    tail_emphasis: float = 1.25


@dataclass
class TrainingConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    hidden_dim: int = 64
    seed: int = 42


@dataclass
class ExperimentConfig:
    datasets_all: list[str] = field(default_factory=list)
    datasets_main: list[str] = field(default_factory=list)
    n_select_main: int = 9
    pilots: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    test_size: float = 0.2
    tail_quantile: float = 0.15
    methods: list[str] = field(default_factory=list)


@dataclass
class TailScoreConfig:
    corruption: CorruptionConfig = field(default_factory=CorruptionConfig)
    tcse: TCSEConfig = field(default_factory=TCSEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


def _merge_dataclass(obj: Any, data: dict[str, Any]) -> None:
    for key, value in data.items():
        if not hasattr(obj, key):
            continue
        current = getattr(obj, key)
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _merge_dataclass(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: str | Path) -> TailScoreConfig:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = TailScoreConfig()
    _merge_dataclass(cfg, raw)
    return cfg


def default_config_path(name: str = "default.yaml") -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / name
