"""Drift: is the plant drifting away from what the model learned?

Two independent signals:
* out-of-time errors: the model, fitted only on earlier runs, mispredicts recent runs
  in one direction;
* the weekly reference run (same standard recipe every week) moves away from its
  earlier level.
Both agreeing gives 근거 강함; one alone gives 보통.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schema import Schema


@dataclass
class DriftSignal:
    kind: str  # "errors" | "reference"
    delta: float  # %p, negative = lower than expected / than before
    z: float
    recent_n: int
    detail: dict

    @property
    def flagged(self) -> bool:
        return abs(self.delta) >= 0.7 and abs(self.z) >= 2.0


def error_drift(errors: pd.DataFrame, days: np.ndarray, window_days: float = 14) -> DriftSignal | None:
    """``errors`` from ``rolling_origin_errors``; ``days`` aligned with its ``row`` column."""
    if errors.empty:
        return None
    day = days[errors["row"].to_numpy()]
    recent = day >= day.max() - window_days
    e = errors["error"].to_numpy()
    if recent.sum() < 3 or (~recent).sum() < 3:
        return None
    delta = e[recent].mean() - e[~recent].mean()
    se = np.sqrt(e[recent].var(ddof=1) / recent.sum() + e[~recent].var(ddof=1) / (~recent).sum())
    weekly = pd.DataFrame({"week": (day // 7).astype(int), "error": e}).groupby("week")["error"].mean()
    return DriftSignal("errors", float(delta), float(delta / se) if se > 0 else 0.0, int(recent.sum()),
                       {"window_days": window_days, "weekly_error": weekly.round(2).to_dict()})


def reference_drift(data: pd.DataFrame, schema: Schema, window_days: float = 21) -> DriftSignal | None:
    col = schema.reference_column
    if not col or col not in data or "days" not in data:
        return None
    ref = data[data[col].astype(bool)].sort_values("days")
    if len(ref) < 6:
        return None
    y = ref[schema.target.name].to_numpy(float)
    d = ref["days"].to_numpy(float)
    recent = d >= d.max() - window_days
    if recent.sum() < 2 or (~recent).sum() < 4:
        return None
    base_mean, base_sd = y[~recent].mean(), y[~recent].std(ddof=1)
    delta = y[recent].mean() - base_mean
    z = delta / (base_sd / np.sqrt(recent.sum())) if base_sd > 0 else 0.0
    return DriftSignal("reference", float(delta), float(z), int(recent.sum()),
                       {"window_days": window_days, "baseline_mean": float(base_mean),
                        "recent_mean": float(y[recent].mean()),
                        "series": [{"days": float(a), "yield": float(b)} for a, b in zip(d, y)]})
