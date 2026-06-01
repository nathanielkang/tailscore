#!/usr/bin/env python
"""Aggregate R6 + flagship experiment status into one JSON with pilot_status."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "r6_experiment_status.json"


def _load(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    synth = _load("r6_synthetic_suite.json") or {}
    decoup = _load("decoupling_analysis.json") or {}
    pilot = _load("pilot_r6_summary.json") or {}
    flagship = _load("flagship_9_summary.json") or {}

    agg = synth.get("aggregate", {})
    c1 = bool(agg.get("clause_1_pass"))
    c2 = bool(agg.get("clause_2_pass"))
    c3 = bool(decoup.get("aggregate", {}).get("clause_3_pass", decoup.get("clause_3_pass")))
    c4 = bool(pilot.get("clause_4", {}).get("clause_4_pass"))
    pilot_pass = c1 and c2 and c3 and c4

    fw = flagship.get("win_vs_tong", {})
    by_ds = fw.get("by_dataset") or {}
    n_wins = int(fw.get("n_wins", 0))
    n_ds = len(by_ds) if by_ds else len(flagship.get("datasets", []))
    flagship_pass = n_wins >= max(8, n_ds - 1) if n_ds else False

    status = {
        "schema": "r6_experiment_status/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pilot_status": "PASS" if pilot_pass else "FAIL",
        "r6_clauses": {
            "clause_1_synthetic_win_rate": c1,
            "clause_2_effect_size": c2,
            "clause_3_decoupling": c3,
            "clause_4_real_pilot": c4,
        },
        "flagship_9": {
            "n_wins_vs_tong": n_wins,
            "n_datasets": n_ds,
            "min_bar": "8/9",
            "flagship_status": "PASS" if flagship_pass else "FAIL",
        },
        "overall_ready_for_manuscript": pilot_pass and flagship_pass,
        "sources": {
            "synthetic": "r6_synthetic_suite.json",
            "decoupling": "decoupling_analysis.json",
            "pilot": "pilot_r6_summary.json",
            "flagship": "flagship_9_summary.json",
        },
    }
    OUT.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))
    return 0 if pilot_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
