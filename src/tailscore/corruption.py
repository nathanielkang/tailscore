"""Train vs eval corruption schedules (A0 — PROPOSAL parity)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CorruptionSchedule:
    """Named corruption mode wired from YAML ``corruption.train`` / ``corruption.eval``."""

    name: str

    def apply_categorical(
        self,
        cat_values: np.ndarray,
        rng: np.random.Generator,
        *,
        phase: str,
    ) -> np.ndarray:
        """Return corrupted categorical column for ``phase`` in {train, eval}."""
        out = cat_values.copy()
        if self.name == "mdm" and phase == "train":
            mask = rng.random(out.shape) < 0.15
            if out.size > 0:
                uniq = np.unique(out[~np.isnan(out.astype(float))]) if out.dtype != object else np.unique(out)
                if uniq.size > 1:
                    repl = rng.choice(uniq, size=int(mask.sum()))
                    out[mask] = repl
        # clean_cat: categoricals stay at clean c0 during eval energy draws
        return out

    def gaussian_noise(
        self,
        values: np.ndarray,
        rng: np.random.Generator,
        sigma: float = 0.1,
    ) -> np.ndarray:
        return values + sigma * rng.standard_normal(values.shape)


def schedule_from_config(train_mode: str, eval_mode: str) -> tuple[CorruptionSchedule, CorruptionSchedule]:
    return CorruptionSchedule(train_mode), CorruptionSchedule(eval_mode)
