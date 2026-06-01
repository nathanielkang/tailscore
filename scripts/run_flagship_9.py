#!/usr/bin/env python
"""9-dataset flagship experiment (selected_main × 3 seeds) — separate from R6 Clause 4 gate."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(ROOT), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("TAILSCORE_OFFLINE", "1")

from tailscore.config import default_config_path, load_config

import run_pilot as pilot_mod


def _load_selected_main() -> list[str]:
    sel_path = ROOT / "results" / "dataset_selection.json"
    if sel_path.exists():
        record = json.loads(sel_path.read_text(encoding="utf-8"))
        selected = record.get("selected_main") or []
        if len(selected) == 9:
            return list(selected)
    raise FileNotFoundError(
        f"Missing or invalid {sel_path.relative_to(ROOT)} — "
        "run: python scripts/run_pilot.py --config configs/default.yaml --select-datasets"
    )


def _win_table(runs: list[dict]) -> dict:
    """Per-dataset majority vote across seeds + aggregate win/loss counts."""
    by_ds: dict[str, list[bool]] = {}
    for r in runs:
        ds = r.get("dataset", "unknown")
        by_ds.setdefault(ds, []).append(bool(r.get("win_vs_tong", False)))

    per_dataset: dict[str, dict] = {}
    for ds, wins_list in sorted(by_ds.items()):
        n_win_seeds = sum(wins_list)
        n_seeds = len(wins_list)
        verdict = n_win_seeds > n_seeds / 2
        per_dataset[ds] = {
            "win": verdict,
            "n_win_seeds": n_win_seeds,
            "n_seeds": n_seeds,
        }

    n_wins = sum(1 for v in per_dataset.values() if v["win"])
    n_losses = len(per_dataset) - n_wins
    return {
        "per_dataset": per_dataset,
        "n_wins": n_wins,
        "n_losses": n_losses,
        "n_datasets": len(per_dataset),
    }


def main() -> int:
    cfg_path = default_config_path("flagship_9.yaml")
    cfg = load_config(cfg_path)
    pilots = list(cfg.experiment.pilots) or _load_selected_main()
    if len(pilots) != 9:
        raise ValueError(f"flagship expects 9 datasets, got {len(pilots)}: {pilots}")

    seeds = list(cfg.experiment.seeds or [cfg.training.seed])
    runs: list[dict] = []
    load_errors: list[dict] = []
    t0 = time.time()
    n_total = len(pilots) * len(seeds)

    print("[TailScore flagship-9]")
    print(f"  config: {cfg_path}")
    print(f"  datasets: {pilots}")
    print(f"  seeds: {seeds}")
    print(f"  T={cfg.tcse.T}")

    for ds in pilots:
        for seed in seeds:
            print(f"[{len(runs)+1}/{n_total}] {ds} seed={seed} ...", flush=True)
            try:
                r = pilot_mod._run_single_pilot(ds, cfg, seed)
                print(
                    f"  tcse={r['tail_mae']:.3f} tong={r['tong_tail_mae']:.3f}"
                    f" win={r['win_vs_tong']}"
                )
                runs.append(r)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                err = {"dataset": ds, "seed": seed, "error": str(exc), "win_vs_tong": False}
                runs.append(err)
                load_errors.append(err)

    win_table = _win_table(runs)
    summary = {
        "schema": "flagship_9_summary/v1",
        "created_at": pilot_mod._utc_now(),
        "status": "flagship_9",
        "config": str(cfg_path),
        "datasets": pilots,
        "corruption": {"train": cfg.corruption.train, "eval": cfg.corruption.eval},
        "score_partition_ratio": cfg.tcse.score_partition_ratio,
        "T": cfg.tcse.T,
        "runs": runs,
        "n_runs": len(runs),
        "n_datasets": 9,
        "win_vs_tong": win_table,
        "load_errors": load_errors,
        "elapsed_sec": round(time.time() - t0, 1),
        "note": (
            "Flagship 9-dataset table run — does not replace pilot_r6_summary.json "
            "(R6 Clause 4 gate remains on 7 pre-registered pilots)."
        ),
    }
    out = ROOT / "results" / "flagship_9_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    print(f"  wins vs Tong: {win_table['n_wins']}/{win_table['n_datasets']}")
    print(f"  per-dataset: {win_table['per_dataset']}")
    if load_errors:
        print(f"  load errors: {len(load_errors)}")
    return 0 if not load_errors else 1


if __name__ == "__main__":
    sys.exit(main())
