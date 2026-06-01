"""Baseline method stubs — equal tuning budget hooks (PROPOSAL §16c)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from tailscore.config import TailScoreConfig
from tailscore.evaluation.metrics import evaluate_predictions
from tailscore.trainer import predict, train_regressor


@dataclass
class BaselineResult:
    method: str
    y_pred: np.ndarray
    weights: Optional[np.ndarray]
    metrics: dict[str, float]
    train_losses: list[float]
    spearman: float | None = None


def _uniform_weights(n: int) -> np.ndarray:
    return np.ones(n, dtype=np.float64)


def _fds_weights(y: np.ndarray, n_bins: int = 20) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).ravel()
    bins = np.linspace(y.min(), y.max(), n_bins + 1)
    idx = np.clip(np.digitize(y, bins) - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts[idx]
    return inv / inv.mean()


def _denseloss_weights(y: np.ndarray) -> np.ndarray:
    from scipy.stats import gaussian_kde

    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 3 or np.std(y) < 1e-12:
        return _uniform_weights(len(y))
    kde = gaussian_kde(y)
    dens = np.clip(kde(y), 1e-8, None)
    inv = 1.0 / dens
    return inv / inv.mean()


def _tong_reg_weights(X: np.ndarray, y: np.ndarray, k: int = 32) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(y))).fit(X)
    dist, idx = nn.kneighbors(X)
    sim = 1.0 / (dist[:, 1:] + 1e-6)
    y_nbr = y[idx[:, 1:]]
    y_center = y.reshape(-1, 1)
    score_sim = np.exp(-np.abs(y_nbr - y_center))
    w = (sim * score_sim).mean(axis=1)
    return w / (w.mean() + 1e-8)


def _prime_t_weights(y: np.ndarray) -> np.ndarray:
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = max(q3 - q1, 1e-6)
    tail = np.abs(y - np.median(y)) > 1.5 * iqr
    w = np.ones_like(y, dtype=np.float64)
    w[tail] = 2.0
    return w


def _fit_weighted(
    method: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TailScoreConfig,
    seed: int,
    weight_fn: Callable[..., np.ndarray],
    *,
    uses_x: bool = False,
) -> BaselineResult:
    weights = weight_fn(X_train, y_train) if uses_x else weight_fn(y_train)
    model, train_result = train_regressor(
        X_train,
        y_train,
        sample_weight=weights,
        epochs=cfg.training.epochs,
        hidden_dim=cfg.training.hidden_dim,
        lr=cfg.training.lr,
        seed=seed,
    )
    y_pred = predict(model, X_test)
    metrics = evaluate_predictions(
        y_test, y_pred, tail_quantile=1.0 - cfg.experiment.tail_quantile
    )
    return BaselineResult(
        method=method,
        y_pred=y_pred,
        weights=weights,
        metrics=metrics,
        train_losses=train_result.losses,
    )


def fit_erm(X_train, y_train, X_test, y_test, cfg, seed=42) -> BaselineResult:
    return _fit_weighted("erm", X_train, y_train, X_test, y_test, cfg, seed, lambda y: _uniform_weights(len(y)))


def fit_uniform_erm(X_train, y_train, X_test, y_test, cfg, seed=42) -> BaselineResult:
    return fit_erm(X_train, y_train, X_test, y_test, cfg, seed)


def fit_fds(X_train, y_train, X_test, y_test, cfg, seed=42) -> BaselineResult:
    return _fit_weighted("fds", X_train, y_train, X_test, y_test, cfg, seed, _fds_weights)


def fit_denseloss(X_train, y_train, X_test, y_test, cfg, seed=42) -> BaselineResult:
    return _fit_weighted("denseloss", X_train, y_train, X_test, y_test, cfg, seed, _denseloss_weights)


def fit_tong_reg(X_train, y_train, X_test, y_test, cfg, seed=42) -> BaselineResult:
    res = _fit_weighted(
        "tong_reg", X_train, y_train, X_test, y_test, cfg, seed, _tong_reg_weights, uses_x=True
    )
    from tailscore.evaluation.metrics import weight_spearman

    res.spearman = weight_spearman(res.weights, np.abs(y_train - np.median(y_train)))
    return res


def fit_prime_t(X_train, y_train, X_test, y_test, cfg, seed=42) -> BaselineResult:
    return _fit_weighted("prime_t", X_train, y_train, X_test, y_test, cfg, seed, _prime_t_weights)


def fit_tabddpm_aug(X_train, y_train, X_test, y_test, cfg, seed=42) -> BaselineResult:
    """Stub: duplicate rare tail rows instead of full TabDDPM synthesis."""
    rng = np.random.default_rng(seed)
    q = np.quantile(y_train, 0.85)
    tail_idx = np.where(y_train >= q)[0]
    if tail_idx.size == 0:
        return fit_erm(X_train, y_train, X_test, y_test, cfg, seed)
    dup = rng.choice(tail_idx, size=min(tail_idx.size, 50), replace=True)
    X_aug = np.vstack([X_train, X_train[dup]])
    y_aug = np.concatenate([y_train, y_train[dup]])
    model, train_result = train_regressor(
        X_aug, y_aug, sample_weight=None,
        epochs=cfg.training.epochs,
        hidden_dim=cfg.training.hidden_dim,
        lr=cfg.training.lr,
        seed=seed,
    )
    y_pred = predict(model, X_test)
    metrics = evaluate_predictions(y_test, y_pred, tail_quantile=1.0 - cfg.experiment.tail_quantile)
    return BaselineResult(
        method="tabddpm_aug",
        y_pred=y_pred,
        weights=None,
        metrics=metrics,
        train_losses=train_result.losses,
    )


BASELINE_REGISTRY = {
    "erm": fit_erm,
    "uniform_erm": fit_uniform_erm,
    "fds": fit_fds,
    "denseloss": fit_denseloss,
    "tong_reg": fit_tong_reg,
    "prime_t": fit_prime_t,
    "tabddpm_aug": fit_tabddpm_aug,
}
