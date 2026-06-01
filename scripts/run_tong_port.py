#!/usr/bin/env python
"""Tong-eval-asymmetry-port — frozen comparator runs (DUEL_EXECUTION_CONTRACT).

Runs the Tong-Reg baseline only, with EXACTLY the same splits, seed set,
regressor class, budget, and report parser as run_r6_synthetic_suite.py.
Non-matched runs are invalid per contract.

The frozen comparator script uses identical dataset generation calls
(same spec + same seed) so splits are deterministically matched.

Usage
-----
    # Dry-run: verify frozen config matches suite
    python scripts/run_tong_port.py --dry-run

    # Quick CPU run (must use same --quick / --n-samples as suite)
    python scripts/run_tong_port.py --quick

    # Full frozen run (same seeds as suite; pass --suite-results to cross-check)
    python scripts/run_tong_port.py --config configs/pilot_r6.yaml \\
        --suite-results results/r6_synthetic_suite.json

Output
------
    results/tong_port.json  (schema: results/schemas/tong_port.schema.json)
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
from tailscore.evaluation.metrics import weight_spearman


# ---------------------------------------------------------------------------
# Tong asymmetry proxy (frozen, must match registry implementation exactly)
# ---------------------------------------------------------------------------

def _tong_asymmetry_proxy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 32,
) -> np.ndarray:
    """Frozen Tong asymmetry proxy weights.

    Returns normalised weights; mirrors tailscore.baselines.registry._tong_reg_weights
    exactly so the comparator is frozen.
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(y_train))).fit(X_train)
    dist, idx = nn.kneighbors(X_train)
    sim = 1.0 / (dist[:, 1:] + 1e-6)
    y_nbr = y_train[idx[:, 1:]]
    y_center = y_train.reshape(-1, 1)
    score_sim = np.exp(-np.abs(y_nbr - y_center))
    w = (sim * score_sim).mean(axis=1)
    return w / (w.mean() + 1e-8)


def _eval_tong_frozen(
    spec: dict,
    cfg: TailScoreConfig,
    seed: int,
    n_samples: int,
) -> dict:
    """Single dataset Tong frozen run."""
    data = make_r6_suite_dataset(
        dependency=spec["dependency"],
        tail_severity=spec["tail_severity"],
        cat_complexity=spec["cat_complexity"],
        n_samples=n_samples,
        seed=seed,
    )
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    # Frozen Tong via registry (identical budget to suite)
    fn = BASELINE_REGISTRY["tong_reg"]
    res = fn(X_tr, y_tr, X_te, y_te, cfg, seed=seed)

    # Asymmetry proxy weights for decoupling analysis
    proxy_weights = _tong_asymmetry_proxy(X_tr, y_tr, k=cfg.tcse.k_neighbors)
    proxy_spearman = weight_spearman(proxy_weights, np.abs(y_tr - np.median(y_tr)))

    return {
        "name": data["name"],
        "dependency": spec["dependency"],
        "tail_severity": spec["tail_severity"],
        "cat_complexity": spec["cat_complexity"],
        "seed": seed,
        "n_train": int(X_tr.shape[0]),
        "tong_tail_mae": float(res.metrics["tail_mae"]),
        "tong_mae": float(res.metrics["mae"]),
        "tong_spearman": float(res.spearman) if res.spearman is not None else None,
        "proxy_spearman": float(proxy_spearman),
        # Store proxy weight summary for decoupling analysis (percentiles, not full array)
        "proxy_weight_pctls": {
            str(p): float(np.percentile(proxy_weights, p))
            for p in [10, 25, 50, 75, 90]
        },
    }


# ---------------------------------------------------------------------------
# Cross-check against suite results
# ---------------------------------------------------------------------------

def _cross_check_splits(suite_path: Path, tong_records: list[dict]) -> list[str]:
    """Warn if tong_port splits diverge from suite (should be empty if seeds match)."""
    warnings = []
    if not suite_path.exists():
        warnings.append(f"Suite results not found at {suite_path}; skipping cross-check.")
        return warnings

    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_map = {
        (r["name"], r["seed"]): r for r in suite.get("datasets", [])
    }
    for rec in tong_records:
        key = (rec["name"], rec["seed"])
        if key not in suite_map:
            warnings.append(f"No matching suite record for {key}")
            continue
        sr = suite_map[key]
        suite_tong = sr.get("results", {}).get("tong_reg", {}).get("tail_mae")
        port_tong = rec.get("tong_tail_mae")
        if suite_tong is not None and port_tong is not None:
            if abs(suite_tong - port_tong) > 1e-6:
                warnings.append(
                    f"{key}: suite tong_tail_mae={suite_tong:.6f} "
                    f"vs port={port_tong:.6f}  delta={abs(suite_tong-port_tong):.2e}"
                )
    return warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _make_quick_cfg() -> TailScoreConfig:
    cfg = load_config(default_config_path("smoke.yaml"))
    cfg.tcse.T = 5
    cfg.tcse.k_neighbors = 8
    cfg.training.epochs = 5
    cfg.training.hidden_dim = 32
    cfg.experiment.tail_quantile = 0.15
    return cfg


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="Tong frozen comparator port")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path("pilot_r6.yaml"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Dataset size (must match run_r6_synthetic_suite.py invocation)",
    )
    parser.add_argument(
        "--seeds", type=str, default=None,
        help="Comma-separated seeds (must match suite)",
    )
    parser.add_argument(
        "--suite-results",
        type=Path,
        default=ROOT / "results" / "r6_synthetic_suite.json",
        help="Path to r6_synthetic_suite.json for split cross-check",
    )
    args = parser.parse_args()

    if args.quick:
        cfg = _make_quick_cfg()
        n_samples = args.n_samples or 300
        seeds = [42]
    else:
        cfg = load_config(args.config)
        n_samples = args.n_samples or 1000
        seeds = (
            [int(s) for s in args.seeds.split(",")]
            if args.seeds
            else (cfg.experiment.seeds or [42])
        )

    print("=" * 65)
    print("Tong-eval-asymmetry-port (frozen comparator)")
    print("=" * 65)
    print(f"  specs:    {len(R6_SUITE_SPECS)} datasets × {len(seeds)} seed(s)")
    print(f"  n_samples: {n_samples}  epochs={cfg.training.epochs}")
    print(f"  k_neighbors: {cfg.tcse.k_neighbors}")

    if args.dry_run:
        print("\n[dry-run] plan shown; no training.")
        return 0

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    records: list[dict] = []
    n_total = len(R6_SUITE_SPECS) * len(seeds)
    done = 0
    for spec in R6_SUITE_SPECS:
        for seed in seeds:
            name = f"r6_{spec['dependency']}_{spec['tail_severity']}_{spec['cat_complexity']}"
            print(f"  [{done+1}/{n_total}] {name}  seed={seed} ...", end=" ", flush=True)
            t_ds = time.time()
            try:
                rec = _eval_tong_frozen(spec, cfg, seed, n_samples)
                print(f"tong_tail_mae={rec['tong_tail_mae']:.3f}  ({time.time()-t_ds:.1f}s)")
                records.append(rec)
            except Exception as exc:
                print(f"ERROR: {exc}")
                records.append({
                    "name": name, "seed": seed, "error": str(exc),
                    "dependency": spec["dependency"],
                    "tail_severity": spec["tail_severity"],
                    "cat_complexity": spec["cat_complexity"],
                })
            done += 1

    # Cross-check splits against suite
    xcheck_warnings = _cross_check_splits(args.suite_results, records)

    payload = {
        "schema": "tong_port/v1",
        "created_at": _utc_now(),
        "config": str(args.config) if not args.quick else "quick",
        "n_samples_per_ds": n_samples,
        "epochs": cfg.training.epochs,
        "k_neighbors": cfg.tcse.k_neighbors,
        "seeds": seeds,
        "n_runs": len(records),
        "elapsed_sec": round(time.time() - t0, 1),
        "cross_check_warnings": xcheck_warnings,
        "datasets": records,
        "note": (
            "Frozen comparator; runs must match r6_synthetic_suite.py "
            "(same specs, seeds, n_samples, training budget)."
        ),
    }

    out = results_dir / "tong_port.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out.relative_to(ROOT)}")
    if xcheck_warnings:
        print(f"[WARN] {len(xcheck_warnings)} split cross-check warnings — see JSON.")
    else:
        print("[OK] split cross-check: no divergence vs suite.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
