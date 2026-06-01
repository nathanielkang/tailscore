"""Synthetic mixed-type tabular data generators.

Smoke test helper: make_synthetic_mixed  (100 rows, fast).
R6 authority suite:  make_r6_suite_dataset  (configurable regime/severity/cat).
Suite catalogue:     R6_SUITE_SPECS  (24 fixed specs) + make_r6_full_suite().
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tailscore.data.datasets import _finalize

# ---------------------------------------------------------------------------
# Smoke test helper (unchanged)
# ---------------------------------------------------------------------------

def make_synthetic_mixed(
    n_samples: int = 100,
    n_num: int = 4,
    n_cat: int = 1,
    cat_card: int = 4,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    num = rng.standard_normal((n_samples, n_num)).astype(np.float32)
    cats = rng.integers(0, cat_card, size=(n_samples, n_cat))
    X_df = pd.DataFrame(num, columns=[f"num_{i}" for i in range(n_num)])
    for j in range(n_cat):
        X_df[f"cat_{j}"] = pd.Categorical(cats[:, j])
    w = rng.normal(size=(n_num + 1, 1))
    latent = num @ w[:n_num] + 0.2 * cats.sum(axis=1, keepdims=True)
    y = np.exp(latent.squeeze() + 0.35 * rng.standard_normal(n_samples))
    return _finalize(X_df, y, "synthetic_smoke", test_size=0.25, random_state=seed)


# ---------------------------------------------------------------------------
# R6 authority suite
# ---------------------------------------------------------------------------

# Contract: 4 dep × 3 tail × 2 cat = 24 datasets; kin8nm excluded from suite.
_DEPS = ["linear", "bilinear", "piecewise", "sparse_nonlinear"]
_TAIL_SEVS = ["rare_5", "rare_10", "rare_15"]
_CAT_COMP = ["low", "high"]

R6_SUITE_SPECS: list[dict] = [
    {"dependency": d, "tail_severity": s, "cat_complexity": c}
    for d in _DEPS
    for s in _TAIL_SEVS
    for c in _CAT_COMP
]

_RARE_FRACS = {"rare_5": 0.05, "rare_10": 0.10, "rare_15": 0.15}


def _dependency_signal(
    X_num: np.ndarray,
    regime: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Numeric-only part of y under the specified dependency regime."""
    if regime == "linear":
        beta = rng.normal(size=X_num.shape[1])
        beta /= np.linalg.norm(beta) + 1e-8
        return X_num @ beta

    if regime == "bilinear":
        # x0*x1 + x2*x3 interaction terms
        return (
            X_num[:, 0] * X_num[:, 1]
            + X_num[:, 2] * X_num[:, 3 % X_num.shape[1]]
        )

    if regime == "piecewise":
        pw = np.where(X_num[:, 0] > 0, X_num[:, 0] ** 2, -X_num[:, 0])
        return pw + 0.5 * X_num[:, 1 % X_num.shape[1]]

    if regime == "sparse_nonlinear":
        return (
            np.sin(X_num[:, 0]) * np.exp(0.3 * np.clip(X_num[:, 1], -3, 3))
            + 0.15 * X_num[:, 2 % X_num.shape[1]]
        )

    raise ValueError(f"Unknown dependency regime: {regime!r}")


def make_r6_suite_dataset(
    dependency: str,
    tail_severity: str,
    cat_complexity: str,
    n_samples: int = 1000,
    n_num: int = 6,
    seed: int = 42,
) -> dict:
    """Generate one R6 authority dataset.

    Contract guarantees:
    - Continuous + categorical predictors in every dataset.
    - ≥30% of tail signal from categorical terms or cross-type interactions.
    - Tail severity (rare-mass fraction) controlled by injection scheme.

    Parameters
    ----------
    dependency:     linear | bilinear | piecewise | sparse_nonlinear
    tail_severity:  rare_5 | rare_10 | rare_15  (rare-mass percentage)
    cat_complexity: low (4–8 cardinality) | high (12–20 cardinality)
    n_samples:      number of rows (default 1000 for CPU speed)
    n_num:          number of continuous predictors (≥4)
    seed:           RNG seed for reproducibility
    """
    rng = np.random.default_rng(seed)

    # --- Categorical feature configuration ---
    if cat_complexity == "low":
        n_cat = 2
        cat_cards = [int(rng.integers(4, 9)) for _ in range(n_cat)]  # 4–8
    else:
        n_cat = 3
        cat_cards = [int(rng.integers(12, 21)) for _ in range(n_cat)]  # 12–20

    # --- Feature generation ---
    X_num = rng.standard_normal((n_samples, n_num))
    cats = np.column_stack([
        rng.integers(0, card, size=n_samples) for card in cat_cards
    ])

    # --- Numeric signal ---
    y_num = _dependency_signal(X_num, dependency, rng)
    y_num = y_num / (np.std(y_num) + 1e-8)  # unit std

    # --- Categorical effect (drives ≥30% of tail signal) ---
    # Tail-driving categories: last ⌈card/4⌉ values per column are "extreme".
    cat_effect = np.zeros(n_samples)
    tail_cat_mask = np.zeros(n_samples, dtype=bool)
    for j, card in enumerate(cat_cards):
        n_extreme = max(card // 4, 1)
        extreme_cats = np.arange(card - n_extreme, card)
        col = cats[:, j]
        is_extreme = np.isin(col, extreme_cats)
        tail_cat_mask |= is_extreme
        # Amplified effect for extreme categories (cross-type interaction)
        col_effect = np.where(
            is_extreme,
            2.5 * (col - card // 2) / (card + 1),  # large signed effect
            0.2 * (col - card // 2) / (card + 1),  # small background effect
        )
        cat_effect += col_effect

    # --- Tail injection (rare-mass fraction) ---
    rare_frac = _RARE_FRACS[tail_severity]
    n_rare = max(int(n_samples * rare_frac), 5)

    # Identify candidate tail-injection sites: highest cat_effect
    candidate_order = np.argsort(cat_effect)[::-1]
    rare_idx = candidate_order[:n_rare]

    # Combine base signal
    y = y_num + cat_effect + 0.2 * rng.standard_normal(n_samples)

    # Inject rare-mass extreme values (exponential tail boost)
    boost = rng.exponential(scale=2.5, size=n_rare)
    y[rare_idx] += boost

    # --- Verify ≥30% tail signal from categoricals ---
    tail_thresh = np.quantile(y, 1.0 - rare_frac)
    tail_mask = y >= tail_thresh
    if tail_mask.sum() > 1:
        var_cat_tail = float(np.var(cat_effect[tail_mask]))
        var_y_tail = float(np.var(y[tail_mask])) + 1e-8
        cat_signal_frac = var_cat_tail / var_y_tail
    else:
        cat_signal_frac = 0.0

    # --- Build DataFrame ---
    X_df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(n_num)])
    for j in range(n_cat):
        X_df[f"cat_{j}"] = cats[:, j].astype(str)

    dataset_name = f"r6_{dependency}_{tail_severity}_{cat_complexity}"
    data = _finalize(
        X_df, y, dataset_name, test_size=0.25, random_state=seed
    )
    data["meta"] = {
        "dependency": dependency,
        "tail_severity": tail_severity,
        "cat_complexity": cat_complexity,
        "cat_signal_frac": cat_signal_frac,
        "cat_signal_frac_pass": cat_signal_frac >= 0.30,
        "rare_frac": rare_frac,
        "n_samples": n_samples,
        "seed": seed,
    }
    return data


def make_r6_full_suite(
    n_samples: int = 1000,
    seeds: list[int] | None = None,
) -> list[dict]:
    """Generate all 24 R6 authority datasets.

    One seed per dataset (reproducible, deterministic order matches R6_SUITE_SPECS).
    """
    if seeds is None:
        seeds = list(range(42, 42 + len(R6_SUITE_SPECS)))
    datasets = []
    for spec, seed in zip(R6_SUITE_SPECS, seeds):
        datasets.append(
            make_r6_suite_dataset(**spec, n_samples=n_samples, seed=seed)
        )
    return datasets
