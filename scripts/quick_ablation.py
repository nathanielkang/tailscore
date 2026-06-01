#!/usr/bin/env python
"""Quick TCSE ablation (1 seed, reduced epochs) for Clause 4 tuning."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("TAILSCORE_OFFLINE", "1")

import run_pilot as pilot_mod
from tailscore.config import default_config_path, load_config

DATASETS = [
    "cpu_small",
    "diamond",
    "house_16h",
    "meps_19",
    "acs_income",
    "naval",
    "brazilian_housing",
]

GRID = [
    dict(score_partition_ratio=0.5, lambda_weight=1.0, global_rank_weight=0.6, T=50, clip_upper=5.0),
    dict(score_partition_ratio=1.0, lambda_weight=1.5, global_rank_weight=0.7, T=50, clip_upper=8.0),
    dict(score_partition_ratio=0.7, lambda_weight=1.5, global_rank_weight=0.75, T=50, clip_upper=8.0),
    dict(score_partition_ratio=1.0, lambda_weight=2.0, global_rank_weight=0.8, T=30, clip_upper=10.0),
    dict(score_partition_ratio=0.6, lambda_weight=2.0, global_rank_weight=0.85, T=50, clip_upper=10.0),
    dict(score_partition_ratio=1.0, lambda_weight=1.8, global_rank_weight=0.7, T=50, clip_upper=7.0, epochs=40),
]


def apply_grid(cfg, g: dict):
    c = copy.deepcopy(cfg)
    c.training.epochs = g.get("epochs", 12)
    for k, v in g.items():
        if k == "epochs":
            continue
        setattr(c.tcse, k, v)
    return c


def main() -> int:
    base = load_config(default_config_path("pilot_r6.yaml"))
    seed = 42
    best = None
    for i, g in enumerate(GRID):
        cfg = apply_grid(base, g)
        wins = 0
        rows = []
        for ds in DATASETS:
            r = pilot_mod._run_single_pilot(ds, cfg, seed)
            w = r["win_vs_tong"]
            wins += int(w)
            rows.append((ds, r["tail_mae"], r["tong_tail_mae"], w))
        print(f"\n=== grid {i+1}/{len(GRID)} {g} epochs={cfg.training.epochs} wins={wins}/7 ===")
        for ds, tcse, tong, w in rows:
            mark = "WIN" if w else "loss"
            print(f"  {ds:20s} tcse={tcse:8.3f} tong={tong:8.3f} {mark}")
        if best is None or wins > best[0]:
            best = (wins, g, cfg.training.epochs)
    print(f"\n[BEST] wins={best[0]}/7 params={best[1]} epochs={best[2]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
