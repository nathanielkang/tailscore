#!/usr/bin/env python
"""TailScore smoke test — 2-epoch synthetic run (workflow §1).

Verifies imports, corruption YAML wiring, TCSE + baseline stubs, loss decrease,
and writes ``results/pilot_smoke.json`` per PROPOSAL schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from tailscore.baselines.registry import BASELINE_REGISTRY
from tailscore.config import default_config_path, load_config
from tailscore.data.synthetic import make_synthetic_mixed
from tailscore.methods.tcse import fit_tcse


PASS = True


def _check(name: str, ok: bool, detail: str = "") -> None:
    global PASS
    tag = "PASS" if ok else "FAIL"
    line = f"  [{tag}] {name}"
    if detail:
        line += f" - {detail}"
    print(line)
    if not ok:
        PASS = False


def main() -> int:
    global PASS

    print("=" * 60)
    print("TailScore (TCSE) Smoke Test")
    print("=" * 60)

    cfg_path = default_config_path("smoke.yaml")
    cfg = load_config(cfg_path)
    _check("corruption.train=mdm", cfg.corruption.train == "mdm", cfg.corruption.train)
    _check("corruption.eval=clean_cat", cfg.corruption.eval == "clean_cat", cfg.corruption.eval)

    data = make_synthetic_mixed(n_samples=100, n_num=4, n_cat=1, seed=42)
    _check("synthetic rows", data["X_train"].shape[0] >= 50, str(data["X_train"].shape))
    _check("output shape train", data["X_train"].shape[1] == data["X_test"].shape[1])

    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    print("\n[1] TCSE stub (2 epochs)")
    tcse_res = fit_tcse(X_train, y_train, X_test, y_test, cfg, seed=42)
    _check("TCSE pred shape", tcse_res.y_pred.shape == y_test.shape)
    _check("TCSE pred finite", bool(np.all(np.isfinite(tcse_res.y_pred))))
    _check("TCSE weights >= 1", float(tcse_res.weights.min()) >= 1.0 - 1e-6)
    _check(
        "loss decreases",
        len(tcse_res.train_losses) >= 2 and tcse_res.train_losses[-1] <= tcse_res.train_losses[0],
        f"{tcse_res.train_losses[0]:.4f} -> {tcse_res.train_losses[-1]:.4f}",
    )

    print("\n[2] Baseline stubs")
    baseline_metrics = {}
    for name in ("erm", "fds", "tong_reg"):
        fn = BASELINE_REGISTRY[name]
        res = fn(X_train, y_train, X_test, y_test, cfg, seed=42)
        _check(f"{name} finite preds", bool(np.all(np.isfinite(res.y_pred))))
        _check(f"{name} tail_mae finite", np.isfinite(res.metrics["tail_mae"]))
        baseline_metrics[name] = res.metrics

    fds_res = BASELINE_REGISTRY["fds"](X_train, y_train, X_test, y_test, cfg, seed=42)
    tong_res = BASELINE_REGISTRY["tong_reg"](X_train, y_train, X_test, y_test, cfg, seed=42)

    # Tong at T=1 proxy: re-run with tcse.T=1 for spearman slot
    cfg_t1 = load_config(cfg_path)
    cfg_t1.tcse.T = 1
    tcse_t1 = fit_tcse(X_train, y_train, X_test, y_test, cfg_t1, seed=42)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "smoke",
        "dataset": "synthetic_smoke",
        "T": cfg.tcse.T,
        "spearman": float(tcse_res.spearman),
        "tail_mae": float(tcse_res.metrics["tail_mae"]),
        "fds_tail_mae": float(fds_res.metrics["tail_mae"]),
        "tong_spearman_T1": float(tcse_t1.spearman if np.isfinite(tcse_t1.spearman) else tong_res.spearman or 0.0),
        "tong_spearman_T50": float(tong_res.spearman or 0.0),
        "plateau_delta": 0.0,
        "loss_epoch1": float(tcse_res.train_losses[0]),
        "loss_epoch2": float(tcse_res.train_losses[-1]),
        "corruption": {
            "train": cfg.corruption.train,
            "eval": cfg.corruption.eval,
        },
        "methods": baseline_metrics,
        "note": "Scaffold smoke — replace with R6 pilot metrics before manuscript claims",
    }
    out_path = results_dir / "pilot_smoke.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[INFO] wrote {out_path.relative_to(ROOT)}")

    print("\n" + "=" * 60)
    if PASS:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)
    return 0 if PASS else 1


if __name__ == "__main__":
    sys.exit(main())
