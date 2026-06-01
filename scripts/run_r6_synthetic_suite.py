#!/usr/bin/env python
"""R6 authority synthetic suite — 24 mixed-type datasets.

Implements DUEL_EXECUTION_CONTRACT §§ Clause 1 + Clause 2 (synthetic):
  Clause 1: TCSE win-rate vs Tong ≥ 70%; bootstrap 95% CI lower bound > 0.
  Clause 2: standardized effect size ≥ 0.30 on ≥ 2 tail-severity groups.

kin8nm is excluded from this suite (sanity-only, contract §).

Usage
-----
    # Quick CPU run (reduced samples + epochs)
    python scripts/run_r6_synthetic_suite.py --quick

    # Full run (T=50, n=1000, 3 seeds)
    python scripts/run_r6_synthetic_suite.py --config configs/pilot_r6.yaml

    # Dry-run: print plan only
    python scripts/run_r6_synthetic_suite.py --dry-run

Output
------
    results/r6_synthetic_suite.json  (schema: results/schemas/r6_synthetic_suite.schema.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from tailscore.baselines.registry import BASELINE_REGISTRY
from tailscore.config import TailScoreConfig, default_config_path, load_config
from tailscore.data.synthetic import R6_SUITE_SPECS, make_r6_suite_dataset
from tailscore.methods.tcse import compute_tcse_weights, fit_tcse


# ---------------------------------------------------------------------------
# Bootstrap win-rate CI (Clause 1)
# ---------------------------------------------------------------------------

def _bootstrap_win_rate_ci(
    wins: list[int],
    n_boot: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    rng = np.random.default_rng(42) if rng is None else rng
    arr = np.asarray(wins, dtype=float)
    n = len(arr)
    boot = [float(rng.choice(arr, size=n, replace=True).mean()) for _ in range(n_boot)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# ---------------------------------------------------------------------------
# Effect size (Clause 2)
# ---------------------------------------------------------------------------

def _paired_cohens_d(baseline: list[float], method: list[float]) -> float:
    """Paired Cohen's d on tail-MAE gains: (baseline - method) / std(diff).

    Positive d => method beats baseline (lower tail-MAE).
    """
    if len(baseline) != len(method) or len(baseline) < 2:
        return float("nan")
    diff = np.asarray(baseline, float) - np.asarray(method, float)
    return float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))


# ---------------------------------------------------------------------------
# Single dataset evaluation
# ---------------------------------------------------------------------------

def _eval_dataset(
    spec: dict,
    cfg: TailScoreConfig,
    seed: int,
    n_samples: int,
    methods: list[str],
) -> dict:
    data = make_r6_suite_dataset(
        dependency=spec["dependency"],
        tail_severity=spec["tail_severity"],
        cat_complexity=spec["cat_complexity"],
        n_samples=n_samples,
        seed=seed,
    )
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    results: dict[str, dict] = {}
    for method in methods:
        if method == "tcse":
            res = fit_tcse(X_tr, y_tr, X_te, y_te, cfg, seed=seed)
            results["tcse"] = {
                "tail_mae": float(res.metrics["tail_mae"]),
                "mae": float(res.metrics["mae"]),
                "spearman": float(res.spearman),
                "stage": res.stage,
            }
        else:
            fn = BASELINE_REGISTRY.get(method)
            if fn is None:
                continue
            res = fn(X_tr, y_tr, X_te, y_te, cfg, seed=seed)
            entry = {
                "tail_mae": float(res.metrics["tail_mae"]),
                "mae": float(res.metrics["mae"]),
            }
            if res.spearman is not None:
                entry["spearman"] = float(res.spearman)
            results[method] = entry

    return {
        "name": data["name"],
        "dependency": spec["dependency"],
        "tail_severity": spec["tail_severity"],
        "cat_complexity": spec["cat_complexity"],
        "seed": seed,
        "meta": data.get("meta", {}),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate(
    run_records: list[dict],
    clause_1_min_win_rate: float = 0.70,
    clause_2_min_effect: float = 0.30,
    clause_2_min_groups: int = 2,
) -> dict:
    """Compute Clause 1 and Clause 2 from per-dataset run records."""
    wins = []           # 1 if TCSE < Tong on this run, else 0
    tail_mae_gain = []  # normalized gain = (tong - tcse) / tong

    sev_tcse: dict[str, list[float]] = {"rare_5": [], "rare_10": [], "rare_15": []}
    sev_tong: dict[str, list[float]] = {"rare_5": [], "rare_10": [], "rare_15": []}

    for rec in run_records:
        r = rec.get("results", {})
        t_tcse = r.get("tcse", {}).get("tail_mae")
        t_tong = r.get("tong_reg", {}).get("tail_mae")
        if t_tcse is None or t_tong is None:
            continue
        win = int(t_tcse < t_tong)
        wins.append(win)
        gain = (t_tong - t_tcse) / (abs(t_tong) + 1e-8)
        tail_mae_gain.append(gain)

        sev = rec.get("tail_severity")
        if sev in sev_tcse:
            sev_tcse[sev].append(t_tcse)
            sev_tong[sev].append(t_tong)

    # Clause 1
    win_rate = float(np.mean(wins)) if wins else 0.0
    ci_lower, ci_upper = _bootstrap_win_rate_ci(wins) if wins else (0.0, 0.0)
    clause_1_pass = bool(win_rate >= clause_1_min_win_rate and ci_lower > 0)

    # Clause 2: Cohen's d by tail-severity group
    effect_sizes: dict[str, float] = {}
    for sev in ("rare_5", "rare_10", "rare_15"):
        d = _paired_cohens_d(sev_tong[sev], sev_tcse[sev])
        effect_sizes[sev] = round(d, 4) if not (d != d) else float("nan")

    n_pass_groups = sum(
        1 for d in effect_sizes.values()
        if not (d != d) and d >= clause_2_min_effect
    )
    clause_2_pass = bool(n_pass_groups >= clause_2_min_groups)

    return {
        "n_runs": len(wins),
        "win_rate": round(win_rate, 4),
        "bootstrap_ci": [round(ci_lower, 4), round(ci_upper, 4)],
        "mean_norm_gain": round(float(np.mean(tail_mae_gain)), 4) if tail_mae_gain else 0.0,
        "clause_1_pass": clause_1_pass,
        "effect_size_by_severity": effect_sizes,
        "clause_2_pass": clause_2_pass,
        "clause_3_pass": "[TBD_DECOUPLING]",
        "clause_4_pass": "[TBD_PILOT]",
        "overall_pass": "[TBD_ALL_CLAUSES]",
        "note": (
            "Clauses 3 and 4 require analyze_decoupling.py and run_pilot.py respectively."
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_quick_cfg() -> TailScoreConfig:
    cfg = load_config(default_config_path("smoke.yaml"))
    cfg.tcse.T = 5
    cfg.tcse.k_neighbors = 8
    cfg.tcse.score_partition_ratio = 0.5
    cfg.training.epochs = 5
    cfg.training.hidden_dim = 32
    cfg.experiment.tail_quantile = 0.15
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="TailScore R6 synthetic suite runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path("pilot_r6.yaml"),
        help="YAML config (default: pilot_r6.yaml)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick CPU run: T=5, epochs=5, n=300 per dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan without training",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override samples per dataset",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds (e.g. '42,43,44')",
    )
    args = parser.parse_args()

    if args.quick:
        cfg = _make_quick_cfg()
        n_samples = args.n_samples or 300
        seeds = [42]
    else:
        cfg = load_config(args.config)
        n_samples = args.n_samples or 1000
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (cfg.experiment.seeds or [42])

    methods = ["tcse", "erm", "fds", "tong_reg"]

    print("=" * 65)
    print("TailScore R6 Synthetic Suite")
    print("=" * 65)
    print(f"  specs:    {len(R6_SUITE_SPECS)} datasets × {len(seeds)} seed(s)")
    print(f"  n_samples: {n_samples}  T={cfg.tcse.T}  epochs={cfg.training.epochs}")
    print(f"  ratio:     score_partition_ratio={cfg.tcse.score_partition_ratio}")
    print(f"  methods:   {methods}")

    if args.dry_run:
        print("\n[dry-run] plan shown; no training executed.")
        for i, spec in enumerate(R6_SUITE_SPECS):
            print(
                f"  {i+1:2d}. r6_{spec['dependency']}_{spec['tail_severity']}_{spec['cat_complexity']}"
            )
        return 0

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    all_runs: list[dict] = []
    n_total = len(R6_SUITE_SPECS) * len(seeds)
    done = 0
    for spec in R6_SUITE_SPECS:
        for seed in seeds:
            name = f"r6_{spec['dependency']}_{spec['tail_severity']}_{spec['cat_complexity']}"
            print(f"  [{done+1}/{n_total}] {name}  seed={seed} ...", end=" ", flush=True)
            t_ds = time.time()
            try:
                rec = _eval_dataset(spec, cfg, seed, n_samples, methods)
                elapsed_ds = time.time() - t_ds
                tcse_m = rec["results"].get("tcse", {}).get("tail_mae", float("nan"))
                tong_m = rec["results"].get("tong_reg", {}).get("tail_mae", float("nan"))
                print(f"tail_mae tcse={tcse_m:.3f} tong={tong_m:.3f}  ({elapsed_ds:.1f}s)")
                all_runs.append(rec)
            except Exception as exc:
                print(f"ERROR: {exc}")
                all_runs.append(
                    {
                        "name": name,
                        "dependency": spec["dependency"],
                        "tail_severity": spec["tail_severity"],
                        "cat_complexity": spec["cat_complexity"],
                        "seed": seed,
                        "error": str(exc),
                        "results": {},
                    }
                )
            done += 1

    agg = _aggregate(all_runs)
    elapsed = time.time() - t0

    payload = {
        "schema": "r6_synthetic_suite/v1",
        "created_at": _utc_now(),
        "config": str(args.config) if not args.quick else "quick",
        "n_samples_per_ds": n_samples,
        "T": cfg.tcse.T,
        "epochs": cfg.training.epochs,
        "score_partition_ratio": cfg.tcse.score_partition_ratio,
        "seeds": seeds,
        "methods": methods,
        "n_runs": len(all_runs),
        "elapsed_sec": round(elapsed, 1),
        "datasets": all_runs,
        "aggregate": agg,
    }

    out = results_dir / "r6_synthetic_suite.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out.relative_to(ROOT)}")

    print("\n--- Clause summary ---")
    print(f"  Clause 1: win_rate={agg['win_rate']:.3f}  CI=[{agg['bootstrap_ci'][0]:.3f}, {agg['bootstrap_ci'][1]:.3f}]  PASS={agg['clause_1_pass']}")
    print(f"  Clause 2: effects={agg['effect_size_by_severity']}  PASS={agg['clause_2_pass']}")
    print(f"  Clause 3: {agg['clause_3_pass']}")
    print(f"  Clause 4: {agg['clause_4_pass']}")
    overall = "FAIL" if not (agg["clause_1_pass"] and agg["clause_2_pass"]) else "[TBD_3_4]"
    print(f"  Overall:  {overall}")
    return 0


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    sys.exit(main())
