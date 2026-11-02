"""Calibration: make predicted ranges honest, measured on time-ordered data.

A model's own uncertainty is not guaranteed to be right, and random splits flatter it
because neighbouring runs share a lot and a week. So we refit on everything before a
cut-off, predict the next block, and repeat (rolling origin). If the 80% ranges cover
fewer than 80% of what happened next, every range is widened by the same factor.
Ranges are only widened, never narrowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .models import ModelFactory, Prediction, Surrogate
from .schema import Schema


def time_order(data: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    for col in (schema.date_column, "days"):
        if col and col in data:
            return data.sort_values(col, kind="stable")
    return data


def rolling_origin_errors(
    factory: ModelFactory, data: pd.DataFrame, schema: Schema, n_blocks: int = 4, min_train: float = 0.5
) -> pd.DataFrame:
    """Out-of-time predictions for the later part of the data."""
    d = time_order(data, schema).reset_index(drop=True)
    n = len(d)
    start = int(n * min_train)
    edges = np.linspace(start, n, n_blocks + 1).astype(int)
    out = []
    y_name = schema.target.name
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        m = factory().fit(d.iloc[:a])
        p = m.predict(d.iloc[a:b])
        out.append(pd.DataFrame({"row": np.arange(a, b), "actual": d[y_name].iloc[a:b].to_numpy(float),
                                 "mean": p.mean, "sd": p.sd, "train_size": a}))
    e = pd.concat(out, ignore_index=True)
    e["error"] = e["actual"] - e["mean"]
    e["z"] = e["error"] / e["sd"]
    return e


def widen_factor(errors: pd.DataFrame, coverage: float = 0.8, max_factor: float = 5.0) -> float:
    needed = np.quantile(np.abs(errors["z"]), coverage) / norm.ppf(0.5 + coverage / 2)
    return float(np.clip(needed, 1.0, max_factor))


@dataclass
class CalibratedModel:
    base: Surrogate
    factor: float

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def schema(self) -> Schema:
        return self.base.schema

    def fit(self, data: pd.DataFrame) -> "CalibratedModel":
        self.base.fit(data)
        return self

    def predict(self, rows: pd.DataFrame) -> Prediction:
        p = self.base.predict(rows)
        return Prediction(p.mean, p.sd * self.factor)


def fit_calibrated(factory: ModelFactory, data: pd.DataFrame, schema: Schema, coverage: float = 0.8):
    """Fit on all data with ranges widened by time-ordered errors. Returns (model, errors)."""
    errors = rolling_origin_errors(factory, data, schema)
    model = CalibratedModel(factory().fit(data), widen_factor(errors, coverage))
    return model, errors
