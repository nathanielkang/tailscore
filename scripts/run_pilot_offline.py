#!/usr/bin/env python
"""Fast offline R6 pilot (7 charter datasets, 3 seeds) for Clause 4 gate."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for p in (str(SRC), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Offline-first — never block on OpenML retries.
os.environ.setdefault("TAILSCORE_OFFLINE", "1")

from tailscore.config import default_config_path, load_config

import run_pilot as pilot_mod


def main() -> int:
    cfg = load_config(default_config_path("pilot_r6.yaml"))
    pilots = list(cfg.experiment.pilots)
    seeds = list(cfg.experiment.seeds or [cfg.training.seed])
    runs = []
    t0 = time.time()
    n_total = len(pilots) * len(seeds)
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
                runs.append(
                    {"dataset": ds, "seed": seed, "error": str(exc), "win_vs_tong": False}
                )

    clause_4 = pilot_mod._eval_clause_4(runs, min_wins=5, total_datasets=len(pilots))
    gate_checks = pilot_mod._check_pilot_gate(cfg, runs)
    summary = {
        "schema": "pilot_r6_summary/v1",
        "created_at": pilot_mod._utc_now(),
        "status": "pilot_offline",
        "config": str(default_config_path("pilot_r6.yaml")),
        "corruption": {"train": cfg.corruption.train, "eval": cfg.corruption.eval},
        "score_partition_ratio": cfg.tcse.score_partition_ratio,
        "T": cfg.tcse.T,
        "runs": runs,
        "n_runs": len(runs),
        "clause_4": clause_4,
        "pilot_gate_checks": gate_checks,
        "overall_verdict": {
            "clause_4_pass": clause_4["clause_4_pass"],
            "clauses_1_2_3_source": "run_r6_synthetic_suite.py + analyze_decoupling.py",
            "note": "Offline pilot — OpenML skipped when pilot_cache / sklearn ARFF present.",
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    out = ROOT / "results" / "pilot_r6_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    print(f"Clause 4: {'PASS' if clause_4['clause_4_pass'] else 'FAIL'}")
    print(f"  wins: {clause_4['n_wins']}/{clause_4['n_datasets_evaluated']}")
    print(f"  verdicts: {clause_4['dataset_verdicts']}")
    return 0 if clause_4["clause_4_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
