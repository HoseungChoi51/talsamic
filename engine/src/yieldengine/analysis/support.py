"""Where do we actually have data? Used to grey out curves and to limit suggestions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..schema import Schema, Variable


def bandwidth(var: Variable, frac: float = 0.05) -> float:
    return var.span() * frac


def support_counts(data: pd.DataFrame, var: Variable, grid: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Number of past runs within ``frac`` of the variable's range of each grid value."""
    x = data[var.name].to_numpy(float)
    return (np.abs(x[None, :] - grid[:, None]) <= bandwidth(var, frac)).sum(axis=1)


def control_distance(schema: Schema, a: pd.DataFrame, b: pd.DataFrame) -> np.ndarray:
    """Distance between recipes over controllable conditions, shape (len(a), len(b)).

    Continuous differences are divided by the variable's range; a different category
    counts as 1. Moving one variable across its whole range gives distance 1.
    """
    d2 = np.zeros((len(a), len(b)))
    for v in schema.controls:
        if v.is_continuous:
            xa = a[v.name].to_numpy(float)[:, None]
            xb = b[v.name].to_numpy(float)[None, :]
            d2 += ((xa - xb) / v.span()) ** 2
        else:
            d2 += (a[v.name].astype(str).to_numpy()[:, None] != b[v.name].astype(str).to_numpy()[None, :])
    return np.sqrt(d2)


def nearest_distance(schema: Schema, cands: pd.DataFrame, data: pd.DataFrame, chunk: int = 2000) -> np.ndarray:
    out = []
    for s in range(0, len(cands), chunk):
        out.append(control_distance(schema, cands.iloc[s:s + chunk], data).min(axis=1))
    return np.concatenate(out) if out else np.zeros(0)
