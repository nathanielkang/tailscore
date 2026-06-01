"""Shared regressor backend (torch optional, sklearn fallback)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class TrainResult:
    losses: list[float]
    backend: str


def _train_sklearn_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray],
    epochs: int,
    hidden_dim: int,
    seed: int,
) -> tuple[object, TrainResult]:
    from sklearn.neural_network import MLPRegressor

    max_iter = max(epochs, 2)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_dim, hidden_dim // 2 or 1),
        max_iter=max_iter,
        random_state=seed,
        early_stopping=False,
        learning_rate_init=1e-3,
    )
    losses: list[float] = []
    # Warm-start loop to expose epoch-wise loss decrease for smoke test.
    partial = MLPRegressor(
        hidden_layer_sizes=(hidden_dim, hidden_dim // 2 or 1),
        max_iter=1,
        warm_start=True,
        random_state=seed,
        learning_rate_init=1e-3,
    )
    for _ in range(epochs):
        partial.fit(X, y, sample_weight=sample_weight)
        pred = partial.predict(X)
        losses.append(float(np.mean((y - pred) ** 2)))
    partial.fit(X, y, sample_weight=sample_weight)
    return partial, TrainResult(losses=losses, backend="sklearn")


def _train_torch_mlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray],
    epochs: int,
    hidden_dim: int,
    lr: float,
    seed: int,
) -> tuple[object, TrainResult]:
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    device = torch.device("cpu")
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, dtype=torch.float32, device=device).view(-1, 1)
    w_t = None
    if sample_weight is not None:
        w_t = torch.as_tensor(sample_weight, dtype=torch.float32, device=device).view(-1, 1)

    net = nn.Sequential(
        nn.Linear(X.shape[1], hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
        nn.ReLU(),
        nn.Linear(max(hidden_dim // 2, 1), 1),
    ).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses: list[float] = []
    for _ in range(epochs):
        opt.zero_grad()
        pred = net(X_t)
        err = (pred - y_t) ** 2
        if w_t is not None:
            err = err * w_t
        loss = err.mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu().item()))
    return net, TrainResult(losses=losses, backend="torch")


def predict(model: object, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(X), dtype=np.float64).ravel()
    import torch

    model.eval()
    with torch.no_grad():
        X_t = torch.as_tensor(X, dtype=torch.float32)
        out = model(X_t).cpu().numpy().ravel()
    return out.astype(np.float64)


def train_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray] = None,
    epochs: int = 2,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
    prefer_torch: bool = True,
) -> tuple[object, TrainResult]:
    if prefer_torch:
        try:
            return _train_torch_mlp(
                X, y,
                sample_weight=sample_weight,
                epochs=epochs,
                hidden_dim=hidden_dim,
                lr=lr,
                seed=seed,
            )
        except ImportError:
            pass
    return _train_sklearn_mlp(
        X, y,
        sample_weight=sample_weight,
        epochs=epochs,
        hidden_dim=hidden_dim,
        seed=seed,
    )
