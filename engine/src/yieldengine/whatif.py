"""바꿔보기: predicted yield for one recipe, and the small curve under each slider.

Each curve moves one condition across its range with everything else held at the
current recipe. Points with no past runs nearby (among runs with the same categorical
settings, e.g. the same catalyst) are marked ``supported=False`` so the UI greys them out.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .analysis.support import support_counts
from .models import Surrogate
from .schema import Schema


def predict_recipe(model: Surrogate, recipe: dict[str, Any], context: dict[str, Any] | None = None) -> dict:
    p = model.predict(pd.DataFrame([{**recipe, **(context or {})}]))
    lo, hi = p.interval(0.8)
    return {"predicted": round(float(p.mean[0]), 1), "low": round(float(lo[0]), 1), "high": round(float(hi[0]), 1)}


def slider_curves(
    model: Surrogate,
    data: pd.DataFrame,
    schema: Schema,
    recipe: dict[str, Any],
    context: dict[str, Any] | None = None,
    n_grid: int = 26,
    min_runs: int = 2,
) -> dict[str, pd.DataFrame]:
    row = {**recipe, **(context or {})}
    cats = [v for v in schema.controls if not v.is_continuous]
    similar = data
    for c in cats:
        similar = similar[similar[c.name].astype(str) == str(recipe[c.name])]

    queries, spans = [], {}
    for v in schema.controls:
        values = np.linspace(v.lower, v.upper, n_grid) if v.is_continuous else np.array(v.categories, dtype=object)
        start = sum(len(q) for q in queries)
        queries.append(pd.DataFrame([{**row, v.name: x} for x in values]))
        spans[v.name] = (start, start + len(values), values)
    allq = pd.concat(queries, ignore_index=True)
    p = model.predict(allq)  # one call for every slider
    lo, hi = p.interval(0.8)

    out = {}
    for v in schema.controls:
        a, b, values = spans[v.name]
        if v.is_continuous:
            counts = support_counts(similar, v, values.astype(float))
        else:
            others = data
            for c in cats:
                if c.name != v.name:
                    others = others[others[c.name].astype(str) == str(recipe[c.name])]
            counts = np.array([(others[v.name].astype(str) == x).sum() for x in values])
        out[v.name] = pd.DataFrame({"value": values, "mean": p.mean[a:b], "low": lo[a:b], "high": hi[a:b],
                                    "n_nearby": counts, "supported": counts >= min_runs})
    return out
