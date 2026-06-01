#!/usr/bin/env python
"""Correlation-decoupling gate (Clause 3) — DUEL_EXECUTION_CONTRACT.

Computes partial Spearman correlations between TailScore weights and
Tong-asymmetry proxies after controlling for rarity and local density.

Contract thresholds (Clause 3):
    median  |partial rho| <= 0.20
    75th-pct |partial rho| <= 0.30

Both must hold for Clause 3 to PASS.

The script re-runs TCSE + Tong on all 24 R6 synthetic datasets (or a quick
subset), extracts per-sample weight vectors, then computes partial correlations
using OLS residuals -> Spearman.

Usage
-----
    # Quick run (T=5, n=200 per dataset, seed=42 only)
    python scripts/analyze_decoupling.py --quick

    # Full run (mirrors pilot_r6.yaml budget, 3 seeds)
    python scripts/analyze_decoupling.py --config configs/pilot_r6.yaml

    # Dry-run
    python scripts/analyze_decoupling.py --dry-run

Output
------
    results/decoupling_analysis.json
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
from scipy.stats import spearmanr

from tailscore.baselines.registry import BASELINE_REGISTRY
from tailscore.config import TailScoreConfig, default_config_path, load_config
from tailscore.data.synthetic import R6_SUITE_SPECS, make_r6_suite_dataset
from tailscore.methods.tcse import compute_tcse_weights


# ---------------------------------------------------------------------------
# Partial correlation via OLS residuals + Spearman
# ---------------------------------------------------------------------------

def _ols_residuals(y: np.ndarray, X_ctrl: np.ndarray) -> np.ndarray:
    """OLS residuals of regressing y on X_ctrl (with intercept)."""
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.column_stack([np.ones(len(y)), np.asarray(X_ctrl, dtype=np.float64)])
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ beta
    except np.linalg.LinAlgError:
        return y - np.mean(y)


def partial_spearman(
    a: np.ndarray,
    b: np.ndarray,
    controls: np.ndarray,
) -> float:
    """Partial Spearman rho(a, b | controls).

    Computed as Spearman correlation of OLS residuals after projecting out
    the control variables.
    """
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_arr = np.asarray(b, dtype=np.float64).ravel()
    ctrl = np.asarray(controls, dtype=np.float64)
    if ctrl.ndim == 1:
        ctrl = ctrl.reshape(-1, 1)

    res_a = _ols_residuals(a_arr, ctrl)
    res_b = _ols_residuals(b_arr, ctrl)

    if len(res_a) < 4:
        return float("nan")
    rho, _ = spearmanr(res_a, res_b)
    return float(rho) if np.isfinite(rho) else float("nan")


# ---------------------------------------------------------------------------
# Rarity and local-density proxies
# ---------------------------------------------------------------------------

def _rarity_proxy(y: np.ndarray) -> np.ndarray:
    """Rarity = absolute deviation from median, percentile-normalized [0,1]."""
    y = np.asarray(y, dtype=np.float64).ravel()
    dev = np.abs(y - np.median(y))
    return (np.argsort(np.argsort(dev)).astype(float)
            / max(len(dev) - 1, 1))


def _local_density_proxy(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Local density proxy: inverse mean kNN distance (higher = denser region)."""
    from sklearn.neighbors import NearestNeighbors

    n = len(X)
    k_eff = min(k + 1, n - 1)
    if k_eff < 2:
        return np.zeros(n)
    nn = NearestNeighbors(n_neighbors=k_eff).fit(X)
    dists, _ = nn.kneighbors(X)
    mean_dist = dists[:, 1:].mean(axis=1) + 1e-8
    inv_dist = 1.0 / mean_dist
    return inv_dist / (inv_dist.max() + 1e-8)


# ---------------------------------------------------------------------------
# Tong asymmetry proxy weights (frozen mirror of registry implementation)
# ---------------------------------------------------------------------------

def _tong_asymmetry_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int = 32,
) -> np.ndarray:
    """Frozen Tong proxy weights — must mirror registry._tong_reg_weights exactly."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(y_train))).fit(X_train)
    dist, idx = nn.kneighbors(X_train)
    sim = 1.0 / (dist[:, 1:] + 1e-6)
    y_nbr = y_train[idx[:, 1:]]
    y_center = y_train.reshape(-1, 1)
    score_sim = np.exp(-np.abs(y_nbr - y_center))
    w = (sim * score_sim).mean(axis=1)
    return w / (w.mean() + 1e-8)


# ---------------------------------------------------------------------------
# Per-dataset decoupling analysis
# ---------------------------------------------------------------------------

def _analyze_one_dataset(
    spec: dict,
    cfg: TailScoreConfig,
    seed: int,
    n_samples: int,
) -> dict:
    """Run TCSE + Tong on one dataset, compute partial rho."""
    data = make_r6_suite_dataset(
        dependency=spec["dependency"],
        tail_severity=spec["tail_severity"],
        cat_complexity=spec["cat_complexity"],
        n_samples=n_samples,
        seed=seed,
    )
    X_tr, y_tr = data["X_train"], data["y_train"]

    # TCSE weights (score partition only -- no test data needed for weights)
    tcse_weights, _, _ = compute_tcse_weights(X_tr, y_tr, cfg, seed=seed)

    # Tong asymmetry proxy weights (frozen)
    tong_weights = _tong_asymmetry_weights(X_tr, y_tr, k=cfg.tcse.k_neighbors)

    # Control covariates: rarity + local density
    rarity = _rarity_proxy(y_tr)
    density = _local_density_proxy(X_tr, k=min(10, len(X_tr) // 5))
    controls = np.column_stack([rarity, density])

    # Partial Spearman rho(tcse_weights, tong_weights | rarity, density)
    prho = partial_spearman(tcse_weights, tong_weights, controls)
    abs_prho = abs(prho) if not (prho != prho) else float("nan")

    # Also record raw Spearman for context
    raw_rho, _ = spearmanr(tcse_weights, tong_weights)
    raw_rho = float(raw_rho) if np.isfinite(raw_rho) else float("nan")

    return {
        "name": data["name"],
        "dependency": spec["dependency"],
        "tail_severity": spec["tail_severity"],
        "cat_complexity": spec["cat_complexity"],
        "seed": seed,
        "n_train": int(X_tr.shape[0]),
        "partial_rho": round(prho, 5) if not (prho != prho) else None,
        "abs_partial_rho": round(abs_prho, 5) if not (abs_prho != abs_prho) else None,
        "raw_spearman_rho": round(raw_rho, 5) if not (raw_rho != raw_rho) else None,
    }


# ---------------------------------------------------------------------------
# Aggregate + PASS/FAIL
# ---------------------------------------------------------------------------

def _aggregate_decoupling(
    records: list[dict],
    threshold_median: float = 0.20,
    threshold_p75: float = 0.30,
) -> dict:
    abs_rhos = [r["abs_partial_rho"] for r in records if r.get("abs_partial_rho") is not None]

    if not abs_rhos:
        return {
            "n_valid": 0,
            "median_abs_partial_rho": None,
            "p75_abs_partial_rho": None,
            "clause_3_pass": False,
            "threshold_median": threshold_median,
            "threshold_p75": threshold_p75,
            "note": "No valid partial rho values computed.",
        }

    arr = np.asarray(abs_rhos)
    med = float(np.median(arr))
    p75 = float(np.percentile(arr, 75))
    clause_pass = bool(med <= threshold_median and p75 <= threshold_p75)

    return {
        "n_valid": len(abs_rhos),
        "median_abs_partial_rho": round(med, 5),
        "p75_abs_partial_rho": round(p75, 5),
        "max_abs_partial_rho": round(float(arr.max()), 5),
        "mean_abs_partial_rho": round(float(arr.mean()), 5),
        "threshold_median": threshold_median,
        "threshold_p75": threshold_p75,
        "clause_3_pass": clause_pass,
        "fail_reason": (
            None if clause_pass else
            (f"median {med:.4f} > {threshold_median}" if med > threshold_median else "")
            + (" AND " if (med > threshold_median and p75 > threshold_p75) else "")
            + (f"p75 {p75:.4f} > {threshold_p75}" if p75 > threshold_p75 else "")
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
    cfg.training.epochs = 3
    cfg.training.hidden_dim = 32
    cfg.experiment.tail_quantile = 0.15
    return cfg


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    parser = argparse.ArgumentParser(description="TailScore Clause-3 decoupling analyzer")
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path("pilot_r6.yaml"),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick run: T=5, n=200, 1 seed",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds",
    )
    parser.add_argument(
        "--threshold-median",
        type=float,
        default=0.20,
        help="Clause-3 median threshold (default 0.20)",
    )
    parser.add_argument(
        "--threshold-p75",
        type=float,
        default=0.30,
        help="Clause-3 75th-pct threshold (default 0.30)",
    )
    args = parser.parse_args()

    if args.quick:
        cfg = _make_quick_cfg()
        n_samples = args.n_samples or 200
        seeds = [42]
    else:
        cfg = load_config(args.config)
        n_samples = args.n_samples or 600
        seeds = (
            [int(s) for s in args.seeds.split(",")]
            if args.seeds
            else (cfg.experiment.seeds or [42])
        )

    print("=" * 65)
    print("TailScore Clause-3 Decoupling Analysis")
    print("=" * 65)
    print(f"  specs:    {len(R6_SUITE_SPECS)} datasets x {len(seeds)} seed(s)")
    print(f"  n_samples: {n_samples}  T={cfg.tcse.T}")
    print(f"  thresholds: median<={args.threshold_median}, p75<={args.threshold_p75}")

    if args.dry_run:
        print("[dry-run] no computation; plan shown.")
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
            print(f"  [{done+1}/{n_total}] {name} seed={seed} ...", end=" ", flush=True)
            try:
                rec = _analyze_one_dataset(spec, cfg, seed, n_samples)
                prho_str = f"{rec['abs_partial_rho']:.4f}" if rec['abs_partial_rho'] is not None else "nan"
                print(f"|partial_rho|={prho_str}")
                records.append(rec)
            except Exception as exc:
                print(f"ERROR: {exc}")
                records.append({
                    "name": name,
                    "dependency": spec["dependency"],
                    "tail_severity": spec["tail_severity"],
                    "cat_complexity": spec["cat_complexity"],
                    "seed": seed,
                    "error": str(exc),
                    "partial_rho": None,
                    "abs_partial_rho": None,
                })
            done += 1

    agg = _aggregate_decoupling(records, args.threshold_median, args.threshold_p75)
    elapsed = time.time() - t0

    payload = {
        "schema": "decoupling_analysis/v1",
        "created_at": _utc_now(),
        "config": str(args.config) if not args.quick else "quick",
        "n_samples_per_ds": n_samples,
        "T": cfg.tcse.T,
        "score_partition_ratio": cfg.tcse.score_partition_ratio,
        "seeds": seeds,
        "n_runs": len(records),
        "elapsed_sec": round(elapsed, 1),
        "aggregate": agg,
        "datasets": records,
    }

    out = results_dir / "decoupling_analysis.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[OK] wrote {out.relative_to(ROOT)}")

    print("\n--- Clause 3 summary ---")
    agg_r = payload["aggregate"]
    med = agg_r.get("median_abs_partial_rho")
    p75 = agg_r.get("p75_abs_partial_rho")
    passed = agg_r.get("clause_3_pass", False)
    print(f"  median |partial rho| = {med}  (threshold <= {args.threshold_median})")
    print(f"  75th-pct |partial rho| = {p75}  (threshold <= {args.threshold_p75})")
    print(f"  Clause 3: {'PASS' if passed else 'FAIL'}")
    if not passed and agg_r.get("fail_reason"):
        print(f"  Reason: {agg_r['fail_reason']}")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
