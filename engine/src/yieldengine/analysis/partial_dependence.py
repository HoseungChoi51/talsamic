"""Partial dependence with the no-data parts marked.

The curve is the model's average predicted yield when one condition is set to each
grid value for every past run. Grid points with fewer than ``min_runs`` past runs
nearby are marked ``supported=False``; the UI greys them out and findings ignore them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import Surrogate
from ..schema import Schema, Variable
from .support import support_counts


def grid_for(var: Variable, n: int = 26) -> np.ndarray:
    return np.linspace(var.lower, var.upper, n)


def partial_dependence(
    model: Surrogate,
    data: pd.DataFrame,
    schema: Schema,
    var: str,
    grid: np.ndarray | None = None,
    at: dict | None = None,
    min_runs: int = 3,
) -> pd.DataFrame:
    """PD of ``var``. ``at`` fixes other conditions, e.g. ``{"catalyst": "B"}``.

    Support is counted among past runs that match ``at`` on categorical conditions.
    """
    v = schema[var]
    grid = grid_for(v) if grid is None else np.asarray(grid, float)
    at = at or {}
    base = data.copy()
    for k, val in at.items():
        base[k] = val
    reps = pd.concat([base.assign(**{var: g}) for g in grid], ignore_index=True)
    p = model.predict(reps)
    n = len(base)
    mean = p.mean.reshape(len(grid), n).mean(axis=1)
    sd = p.sd.reshape(len(grid), n).mean(axis=1)

    match = data
    for k, val in at.items():
        if not schema[k].is_continuous:
            match = match[match[k].astype(str) == str(val)]
    counts = support_counts(match, v, grid)
    return pd.DataFrame({"value": grid, "mean": mean, "lo": mean - 1.2816 * sd, "hi": mean + 1.2816 * sd,
                         "n_nearby": counts, "supported": counts >= min_runs})


def effect_size(pd_frame: pd.DataFrame) -> float:
    """How much the curve moves yield inside the supported range (max minus min, %p)."""
    s = pd_frame[pd_frame["supported"]]
    return float(s["mean"].max() - s["mean"].min()) if len(s) >= 2 else 0.0
