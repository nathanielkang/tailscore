"""Regression metrics for TailScore pilots."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def tail_mask(y: np.ndarray, quantile: float = 0.85) -> np.ndarray:
    y = np.asarray(y).ravel()
    thresh = np.quantile(y, quantile)
    return y >= thresh


def tail_mae(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.85) -> float:
    mask = tail_mask(y_true, quantile)
    if mask.sum() == 0:
        return float("nan")
    return mae(y_true[mask], y_pred[mask])


def weight_spearman(weights: np.ndarray, severity: np.ndarray) -> float:
    if len(weights) < 3:
        return float("nan")
    rho, _ = spearmanr(weights, severity)
    return float(rho) if np.isfinite(rho) else float("nan")


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tail_quantile: float = 0.85,
) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "tail_mae": tail_mae(y_true, y_pred, tail_quantile),
    }
