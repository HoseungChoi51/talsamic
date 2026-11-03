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


CATEGORY_SWAP = 0.25  # distance of switching one categorical condition (catalyst, ligand, solvent)


def control_distance(schema: Schema, a: pd.DataFrame, b: pd.DataFrame, swap: float = CATEGORY_SWAP) -> np.ndarray:
    """Distance between recipes over controllable conditions, shape (len(a), len(b)).

    Continuous differences are divided by the variable's range; a different category
    counts as ``swap`` (0.25, like moving a number a quarter of its range): one swap fits
    the 균형 risk limit (0.30), two fit 도전적으로 (0.60), none fit 안전하게 (0.15).
    Moving one number across its whole range gives distance 1.
    """
    d2 = np.zeros((len(a), len(b)))
    for v in schema.controls:
        if v.is_continuous:
            xa = a[v.name].to_numpy(float)[:, None]
            xb = b[v.name].to_numpy(float)[None, :]
            d2 += ((xa - xb) / v.span()) ** 2
        else:
            d2 += swap**2 * (a[v.name].astype(str).to_numpy()[:, None] != b[v.name].astype(str).to_numpy()[None, :])
    return np.sqrt(d2)


def nearest_distance(schema: Schema, cands: pd.DataFrame, data: pd.DataFrame, chunk: int = 2000) -> np.ndarray:
    out = []
    for s in range(0, len(cands), chunk):
        out.append(control_distance(schema, cands.iloc[s:s + chunk], data).min(axis=1))
    return np.concatenate(out) if out else np.zeros(0)
