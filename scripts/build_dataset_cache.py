#!/usr/bin/env python
"""Export offline pilot CSVs from local sklearn ARFF cache + charter surrogates.

Run once when online is unavailable but sklearn cache exists:
    python scripts/build_dataset_cache.py
"""

from __future__ import annotations

import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "data" / "pilot_cache"
SKLEARN_DL = (
    Path.home()
    / "scikit_learn_data"
    / "openml"
    / "openml.org"
    / "data"
    / "v1"
    / "download"
)

ARFF_SOURCES: dict[str, list[Path]] = {
    "cpu_small": [
        SKLEARN_DL / "52751" / "cpu_act.arff.gz",
        SKLEARN_DL / "53295" / "cpu_act.arff.gz",
    ],
    "house_16h": [SKLEARN_DL / "52752" / "house_16H.arff.gz"],
    "meps_19": [SKLEARN_DL / "22120800" / "medical_cost.arff.gz"],
    "abalone": [
        SKLEARN_DL / "22111820" / "abalone.arff.gz",
        SKLEARN_DL / "3620" / "abalone.arff.gz",
    ],
    "kin8nm": [SKLEARN_DL / "3626" / "kin8nm.arff.gz"],
}

# Optional online OpenML export (skipped when CSV already exists).
OPENML_FETCH: dict[str, dict] = {}

CHARTER_SURROGATES: dict[str, dict] = {
    "diamond": dict(n_samples=5000, n_num=6, n_cat=3, cat_card=5, seed=1197),
    "acs_income": dict(n_samples=6000, n_num=10, n_cat=4, cat_card=8, seed=42178),
    "naval": dict(n_samples=8000, n_num=12, n_cat=0, cat_card=0, seed=151),
    "brazilian_housing": dict(n_samples=7000, n_num=6, n_cat=4, cat_card=5, seed=42688),
    # Offline fallbacks when OpenML ARFF is unavailable (space_ga/yacht).
    "space_ga": dict(n_samples=6000, n_num=8, n_cat=2, cat_card=6, seed=201),
    "yacht": dict(n_samples=308, n_num=6, n_cat=0, cat_card=0, seed=198),
}


def _load_arff_df(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data, meta = arff.loadarff(fh)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")
    return df


def _save_csv(name: str, df: pd.DataFrame, target: str, source: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{name}.csv"
    meta = OUT / f"{name}.meta.json"
    df.to_csv(out, index=False)
    meta.write_text(
        __import__("json").dumps({"source": source, "target": target, "rows": len(df)}),
        encoding="utf-8",
    )
    print(f"[OK] {name}: {len(df)} rows -> {out.relative_to(ROOT)} ({source})")
    return out


def _export_arff(name: str, paths: list[Path]) -> bool:
    for path in paths:
        if not path.exists():
            continue
        df = _load_arff_df(path)
        target = df.columns[-1]
        return _save_csv(name, df, target, f"sklearn_arff:{path.parent.name}") is not None
    print(f"[SKIP] {name}: no ARFF cache found")
    return False


def _export_surrogate(name: str, spec: dict) -> None:
    from tailscore.data.datasets import _charter_surrogate_frame

    df, target = _charter_surrogate_frame(name, **spec)
    _save_csv(name, df, target, f"charter_surrogate:seed={spec['seed']}")


def _export_openml(name: str, spec: dict) -> bool:
    out = OUT / f"{name}.csv"
    if out.exists():
        print(f"[SKIP] {name}: cache already at {out.relative_to(ROOT)}")
        return True
    from sklearn.datasets import fetch_openml

    data_id = spec.get("data_id")
    openml_name = spec.get("name")
    if data_id is not None:
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    elif openml_name:
        bunch = fetch_openml(openml_name, version=1, as_frame=True, parser="auto")
    else:
        print(f"[SKIP] {name}: no OpenML id/name in spec")
        return False
    frame = bunch.frame.copy()
    target = bunch.target_names[0] if bunch.target_names else bunch.target.name
    _save_csv(name, frame, target, f"openml:{data_id or openml_name}")
    return True


def _export_california_housing() -> None:
    name = "california_housing"
    out = OUT / f"{name}.csv"
    if out.exists():
        print(f"[SKIP] {name}: cache already at {out.relative_to(ROOT)}")
        return
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df["target"] = data.target.values
    _save_csv(name, df, "target", "sklearn:fetch_california_housing")


def main() -> int:
    for name, paths in ARFF_SOURCES.items():
        _export_arff(name, paths)
    for name, spec in CHARTER_SURROGATES.items():
        _export_surrogate(name, spec)
    for name, spec in OPENML_FETCH.items():
        _export_openml(name, spec)
    _export_california_housing()
    print(f"[DONE] pilot cache at {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
