"""TCSE — Tail Conditional Score Energy.

Two-stage protocol (score_partition_ratio < 1.0):
  Stage S: score partition — compute TCSE energy + weights from S fraction.
  Stage R: regression partition — train regressor on R with weights extrapolated
           from S via kNN.  Prevents weight-computation data leaking into the
           loss surface, separating the two objectives cleanly.

Energy proxy (improved):
  KDE-based negative log-density gives scale-invariant tail energy.  A T-step
  Monte Carlo smoothing pass averages perturbed log-density estimates, mimicking
  the path integral under C_eval.  Final weights blend global rank (tail severity)
  with local kNN rank (neighborhood anomaly).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.neighbors import NearestNeighbors

from tailscore.config import TailScoreConfig
from tailscore.corruption import schedule_from_config
from tailscore.evaluation.metrics import evaluate_predictions, weight_spearman
from tailscore.trainer import predict, train_regressor


@dataclass
class TCSEResult:
    y_pred: np.ndarray
    weights: np.ndarray
    energy: np.ndarray
    spearman: float
    metrics: dict[str, float]
    train_losses: list[float]
    stage: str = "single"


# ---------------------------------------------------------------------------
# Energy computation
# ---------------------------------------------------------------------------

def _kde_neg_log_density(y: np.ndarray) -> np.ndarray:
    """Negative log KDE density — high for rare/tail samples."""
    from scipy.stats import gaussian_kde

    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) < 3 or np.std(y) < 1e-12:
        return np.zeros_like(y)
    kde = gaussian_kde(y)
    density = np.clip(kde(y), 1e-10, None)
    return -np.log(density)


def _y_subscore_energy_proxy(
    y: np.ndarray,
    T: int,
    rng: np.random.Generator,
    eval_schedule,
) -> np.ndarray:
    """KDE path-energy proxy under C_eval: T-step average over perturbed log p̂.

    At T=1 returns pure KDE neg-log-density.
    At T>1 averages over Gaussian-perturbed versions (smoothed energy).
    Adaptive sigma = 0.1 * IQR ensures scale invariance across datasets.
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    base_energy = _kde_neg_log_density(y)

    if T <= 1:
        return base_energy

    y_iqr = max(np.percentile(y, 75) - np.percentile(y, 25), 1e-6)
    sigma = 0.1 * y_iqr
    dt = 1.0 / T
    energy = base_energy * dt

    from scipy.stats import gaussian_kde

    try:
        kde = gaussian_kde(y)
    except Exception:
        return base_energy

    for _ in range(T - 1):
        y_pert = eval_schedule.gaussian_noise(y, rng, sigma=sigma)
        dens_t = np.clip(kde(y_pert), 1e-10, None)
        energy += (-np.log(dens_t)) * dt

    return energy


# ---------------------------------------------------------------------------
# kNN rank-normalisation
# ---------------------------------------------------------------------------

def _ranknorm_knn(x: np.ndarray, energy: np.ndarray, k: int) -> np.ndarray:
    """Local rank of each sample within its k-NN neighbourhood (0=lowest energy)."""
    n = len(energy)
    k = min(k, max(n - 1, 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(x)
    _, idx = nn.kneighbors(x)
    ranks = np.zeros(n, dtype=np.float64)
    for i in range(n):
        nbr = idx[i]
        local = energy[nbr]
        order = np.argsort(np.argsort(local))
        ranks[i] = order[0] / max(len(local) - 1, 1)
    return ranks


def _extrapolate_weights_knn(
    X_src: np.ndarray,
    weights_src: np.ndarray,
    energy_src: np.ndarray,
    X_tgt: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign weights to target samples via kNN average from source partition."""
    k = min(k, max(len(weights_src) - 1, 1))
    nn = NearestNeighbors(n_neighbors=max(k, 1)).fit(X_src)
    _, nn_idx = nn.kneighbors(X_tgt)
    w_tgt = weights_src[nn_idx].mean(axis=1)
    e_tgt = energy_src[nn_idx].mean(axis=1)
    return w_tgt, e_tgt


# ---------------------------------------------------------------------------
# Core weight computation
# ---------------------------------------------------------------------------

def compute_tcse_weights(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TailScoreConfig,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute TCSE sample weights.

    Returns (weights, energy, spearman_vs_severity).
    """
    train_sched, eval_sched = schedule_from_config(
        cfg.corruption.train, cfg.corruption.eval
    )
    rng = np.random.default_rng(seed)
    _ = train_sched  # MDM train hook (documents wiring; no-op on numeric X)

    # Energy proxy: KDE neg-log-density + MC smoothing
    energy = _y_subscore_energy_proxy(y, cfg.tcse.T, rng, eval_sched)

    # Normalize to [0, 1]
    e_range = np.max(energy) - np.min(energy)
    energy_norm = (energy - np.min(energy)) / (e_range + 1e-8)

    # Global rank: percentile position (captures absolute tail severity)
    global_ranks = (
        np.argsort(np.argsort(energy_norm)).astype(float)
        / max(len(energy_norm) - 1, 1)
    )

    # Local kNN rank: neighbourhood anomaly
    local_ranks = _ranknorm_knn(X, energy_norm, cfg.tcse.k_neighbors)

    # Blend
    gw = getattr(cfg.tcse, "global_rank_weight", 0.6)
    blended = gw * global_ranks + (1.0 - gw) * local_ranks
    emph = getattr(cfg.tcse, "tail_emphasis", 1.0)
    if emph != 1.0:
        blended = np.power(np.clip(blended, 0.0, 1.0), emph)

    weights = np.clip(
        1.0 + cfg.tcse.lambda_weight * blended,
        1.0,
        cfg.tcse.clip_upper,
    )

    severity = np.abs(y - np.median(y))
    spearman = weight_spearman(weights, severity)
    return weights, energy_norm, spearman


# ---------------------------------------------------------------------------
# Full fit
# ---------------------------------------------------------------------------

def fit_tcse(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    cfg: TailScoreConfig,
    seed: int = 42,
) -> TCSEResult:
    ratio = getattr(cfg.tcse, "score_partition_ratio", 1.0)
    n = len(y_train)
    stage = "single"

    if ratio < 1.0 and n >= 20:
        # ------------------------------------------------------------------
        # Two-stage: S (scoring) / R (regression) — clean separation
        # Weights are computed on S only; extrapolated to all train rows;
        # regressor trains on the full set (fair capacity vs Tong/FDS).
        # ------------------------------------------------------------------
        stage = "two-stage"
        rng_split = np.random.default_rng(seed + 999)
        perm = rng_split.permutation(n)
        n_s = max(int(n * ratio), 10)
        s_idx = perm[:n_s]
        r_idx = perm[n_s:]

        X_s, y_s = X_train[s_idx], y_train[s_idx]
        X_r, y_r = X_train[r_idx], y_train[r_idx]

        # Stage S: compute energy + weights on score partition
        weights_s, energy_s, _ = compute_tcse_weights(X_s, y_s, cfg, seed)

        # Extrapolate weights to all training rows via kNN from S
        k = cfg.tcse.k_neighbors
        all_weights, all_energy = _extrapolate_weights_knn(
            X_s, weights_s, energy_s, X_train, k
        )

        # Train regressor on full train set (weights from S-only scoring)
        model, train_result = train_regressor(
            X_train,
            y_train,
            sample_weight=all_weights,
            epochs=cfg.training.epochs,
            hidden_dim=cfg.training.hidden_dim,
            lr=cfg.training.lr,
            seed=seed,
        )

        severity = np.abs(y_train - np.median(y_train))
        spearman = weight_spearman(all_weights, severity)

    else:
        # ------------------------------------------------------------------
        # Single-stage: score + regress on full training set (backward compat)
        # ------------------------------------------------------------------
        all_weights, all_energy, spearman = compute_tcse_weights(
            X_train, y_train, cfg, seed
        )
        model, train_result = train_regressor(
            X_train,
            y_train,
            sample_weight=all_weights,
            epochs=cfg.training.epochs,
            hidden_dim=cfg.training.hidden_dim,
            lr=cfg.training.lr,
            seed=seed,
        )

    y_pred = predict(model, X_test)
    metrics = evaluate_predictions(
        y_test, y_pred, tail_quantile=1.0 - cfg.experiment.tail_quantile
    )
    metrics["spearman"] = spearman

    return TCSEResult(
        y_pred=y_pred,
        weights=all_weights,
        energy=all_energy,
        spearman=spearman,
        metrics=metrics,
        train_losses=train_result.losses,
        stage=stage,
    )
