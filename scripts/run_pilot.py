#!/usr/bin/env python
"""Round 6 pilot runner — linked gate at T=50 (DUEL_EXECUTION_CONTRACT Clause 4).

Usage:
    python scripts/run_pilot.py --config configs/pilot_r6.yaml
    python scripts/run_pilot.py --dry-run
    python scripts/run_pilot.py --select-datasets

Output: results/pilot_r6_summary.json  (schema: results/schemas/pilot_r6_summary.schema.json)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Offline-first unless explicitly disabled (avoids OpenML hang).
os.environ.setdefault("TAILSCORE_OFFLINE", "1")

import numpy as np

from tailscore.baselines.registry import BASELINE_REGISTRY
from tailscore.config import default_config_path, load_config
from tailscore.data.datasets import load_dataset
from tailscore.data.synthetic import make_synthetic_mixed
from tailscore.data.selection import run_selection_protocol, write_selection_json
from tailscore.methods.tcse import fit_tcse


# ---------------------------------------------------------------------------
# Single dataset pilot run
# ---------------------------------------------------------------------------

def _load_pilot_data(dataset: str, cfg, seed: int) -> dict:
    key = dataset.lower().replace("-", "_")
    if key == "synthetic_smoke":
        return make_synthetic_mixed(n_samples=400, seed=seed)
    return load_dataset(dataset, test_size=cfg.experiment.test_size, random_state=seed)


def _run_single_pilot(dataset: str, cfg, seed: int) -> dict:
    data = _load_pilot_data(dataset, cfg, seed)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    tcse = fit_tcse(X_train, y_train, X_test, y_test, cfg, seed=seed)
    fds = BASELINE_REGISTRY["fds"](X_train, y_train, X_test, y_test, cfg, seed=seed)
    tong = BASELINE_REGISTRY["tong_reg"](X_train, y_train, X_test, y_test, cfg, seed=seed)

    # T=1 reference run (plateau gate)
    cfg_t1 = load_config(default_config_path("pilot_r6.yaml"))
    cfg_t1.tcse.T = 1
    cfg_t1.training.epochs = cfg.training.epochs
    tcse_t1 = fit_tcse(X_train, y_train, X_test, y_test, cfg_t1, seed=seed)

    # Win = TCSE beats Tong on tail_mae (strict improvement)
    tcse_tail = float(tcse.metrics["tail_mae"])
    tong_tail = float(tong.metrics["tail_mae"])
    win = bool(tcse_tail < tong_tail)
    tail_mae_gain = (tong_tail - tcse_tail) / (abs(tong_tail) + 1e-8)

    return {
        "dataset": dataset,
        "seed": seed,
        "T": int(cfg.tcse.T),
        "stage": tcse.stage,
        "spearman": float(tcse.spearman),
        "tail_mae": tcse_tail,
        "fds_tail_mae": float(fds.metrics["tail_mae"]),
        "tong_tail_mae": tong_tail,
        "tong_spearman_T1": float(tcse_t1.spearman),
        "tong_spearman_T50": float(tong.spearman) if tong.spearman is not None else None,
        "tail_mae_gain_vs_tong": round(tail_mae_gain, 5),
        "win_vs_tong": win,
        "plateau_delta": None,  # requires T=100 run; set by downstream script
        "status": "pilot",
    }


# ---------------------------------------------------------------------------
# Clause 4 aggregation
# ---------------------------------------------------------------------------

def _eval_clause_4(
    runs: list[dict],
    min_wins: int = 5,
    total_datasets: int = 7,
) -> dict:
    """Clause 4: >= 5/7 wins on pre-registered pilot datasets.

    A win is: TCSE tail_mae < Tong tail_mae on a given (dataset, seed) run.
    Win is counted per unique dataset (majority vote across seeds).
    """
    # Group by dataset; win if majority of seeds show improvement
    dataset_wins: dict[str, list[bool]] = {}
    for r in runs:
        ds = r.get("dataset", "unknown")
        w = r.get("win_vs_tong", False)
        dataset_wins.setdefault(ds, []).append(w)

    dataset_verdicts: dict[str, bool] = {}
    for ds, wins_list in dataset_wins.items():
        dataset_verdicts[ds] = bool(sum(wins_list) > len(wins_list) / 2)

    n_wins = sum(dataset_verdicts.values())
    n_datasets = len(dataset_verdicts)
    clause_4_pass = bool(n_wins >= min_wins)

    return {
        "n_datasets_evaluated": n_datasets,
        "n_wins": n_wins,
        "min_wins_required": min_wins,
        "win_target": f"{min_wins}/{total_datasets}",
        "dataset_verdicts": dataset_verdicts,
        "clause_4_pass": clause_4_pass,
    }


# ---------------------------------------------------------------------------
# Spearman gate check (pilot gate from pilot_r6.yaml)
# ---------------------------------------------------------------------------

def _check_pilot_gate(cfg, runs: list[dict]) -> dict:
    gate = getattr(cfg, "pilot_gate", None)
    results = {}

    # spearman_min
    spearman_min = 0.3
    if gate and hasattr(gate, "spearman_min"):
        spearman_min = gate.spearman_min

    spearmans = [r["spearman"] for r in runs if r.get("spearman") is not None and np.isfinite(r["spearman"])]
    med_sp = float(np.median(spearmans)) if spearmans else float("nan")
    results["spearman_gate"] = {
        "median_spearman": round(med_sp, 4),
        "threshold": spearman_min,
        "pass": bool(np.isfinite(med_sp) and med_sp >= spearman_min),
    }

    # n_tail_min: average n_test across runs >= 50
    results["n_tail_check"] = {
        "note": "Requires n_test >= 50 tail samples; verified at dataset-load time."
    }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="TailScore R6 pilot runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path("pilot_r6.yaml"),
        help="Pilot YAML config",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--select-datasets", action="store_true", help="Run 12->9 selection stub")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pilots = cfg.experiment.pilots
    seeds = cfg.experiment.seeds or [cfg.training.seed]

    print("[TailScore R6 pilot]")
    print(f"  config: {args.config}")
    print(f"  corruption.train={cfg.corruption.train} eval={cfg.corruption.eval}")
    print(f"  score_partition_ratio={cfg.tcse.score_partition_ratio}")
    print(f"  pilots: {pilots}")
    print(f"  seeds: {seeds}")
    print(f"  T={cfg.tcse.T}")

    if args.dry_run:
        print("[dry-run] no training executed")
        return 0

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.select_datasets:
        record = run_selection_protocol(n_select=cfg.experiment.n_select_main, seed=cfg.training.seed)
        sel_path = write_selection_json(results_dir / "dataset_selection.json", record)
        print(f"[OK] wrote {sel_path}")
        return 0

    all_runs: list[dict] = []
    t0 = time.time()
    n_total = len(pilots) * len(seeds)
    done = 0
    for ds in pilots:
        for seed in seeds:
            print(f"  [{done+1}/{n_total}] {ds} seed={seed} ...", end=" ", flush=True)
            try:
                result = _run_single_pilot(ds, cfg, seed)
                print(
                    f"tail_mae={result['tail_mae']:.3f}"
                    f"  tong={result['tong_tail_mae']:.3f}"
                    f"  win={result['win_vs_tong']}"
                )
                all_runs.append(result)
            except Exception as exc:
                print(f"ERROR: {exc}")
                all_runs.append({
                    "dataset": ds,
                    "seed": seed,
                    "error": str(exc),
                    "status": "error",
                    "win_vs_tong": False,
                })
            done += 1

    # Clause 4 evaluation
    clause_4 = _eval_clause_4(all_runs, min_wins=5, total_datasets=len(pilots))

    # Pilot gate checks
    gate_checks = _check_pilot_gate(cfg, all_runs)

    # Aggregate tail_mae gains
    gains = [r.get("tail_mae_gain_vs_tong") for r in all_runs if r.get("tail_mae_gain_vs_tong") is not None]
    mean_gain = round(float(np.mean(gains)), 5) if gains else None

    summary = {
        "schema": "pilot_r6_summary/v1",
        "created_at": _utc_now(),
        "status": "pilot",
        "config": str(args.config),
        "corruption": {"train": cfg.corruption.train, "eval": cfg.corruption.eval},
        "score_partition_ratio": cfg.tcse.score_partition_ratio,
        "T": cfg.tcse.T,
        "runs": all_runs,
        "n_runs": len(all_runs),
        "elapsed_sec": round(time.time() - t0, 1),
        "mean_tail_mae_gain_vs_tong": mean_gain,
        "clause_4": clause_4,
        "pilot_gate_checks": gate_checks,
        "overall_verdict": {
            "clause_4_pass": clause_4["clause_4_pass"],
            "clauses_1_2_3_source": "run_r6_synthetic_suite.py + analyze_decoupling.py",
            "note": (
                "All 4 clauses must PASS before manuscript claims are unblocked. "
                "Clauses 1-3 require running the R6 synthetic suite and decoupling analysis."
            ),
        },
    }

    out = results_dir / "pilot_r6_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out.relative_to(ROOT)}")

    print("\n--- Clause 4 summary ---")
    print(f"  wins: {clause_4['n_wins']}/{clause_4['n_datasets_evaluated']}"
          f"  (need {clause_4['min_wins_required']})")
    print(f"  dataset verdicts: {clause_4['dataset_verdicts']}")
    print(f"  Clause 4: {'PASS' if clause_4['clause_4_pass'] else 'FAIL'}")

    return 0 if clause_4["clause_4_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
