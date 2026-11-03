"""Calibration: make predicted ranges honest, measured on time-ordered data.

A model's own uncertainty is not guaranteed to be right, and random splits flatter it
because neighbouring runs share a lot and a week. So we refit on everything before a
cut-off, predict the next block, and repeat (rolling origin). If the 80% ranges cover
fewer than 80% of what happened next, every range is widened by the same factor.
Ranges are only widened, never narrowed.

Drift: when the plant shifts (fouling, a new raw material), a model learns the new level
only near the recipes run recently, so new recipes are predicted at the old level. The
final model therefore checks the last two weeks: a model fitted only on the runs before
them predicts them, and if they sit clearly off (at least 0.7%p and two standard errors),
the recent yields are moved back by that amount for fitting and every prediction is moved
by it instead. The range adds the uncertainty of that shift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .korean import fmt_pp
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


def run_days(data: pd.DataFrame, schema: Schema) -> np.ndarray | None:
    """Days since the first run, from ``days`` or the date column; None if neither exists."""
    if "days" in data:
        return data["days"].to_numpy(float)
    col = schema.date_column
    if col and col in data:
        t = pd.to_datetime(data[col])
        return ((t - t.min()) / pd.Timedelta(days=1)).to_numpy(float)
    return None


@dataclass
class DriftShift:
    """Recent runs against a model fitted only on the runs before them (actual − predicted)."""

    delta: float  # %p; negative = recent runs lower than the model expects
    se: float
    n: int
    window_days: float
    applied: bool

    @property
    def note(self) -> str | None:
        """A sentence for the UI and the batch notes when the shift is applied."""
        if not self.applied:
            return None
        word = "낮게" if self.delta < 0 else "높게"
        move = "낮췄다" if self.delta < 0 else "높였다"
        return (f"최근 {int(self.window_days // 7)}주 실험이 그 전 데이터로 예측한 값보다 평균 {fmt_pp(self.delta)} "
                f"{word} 나와서, 모든 예상 수율을 {fmt_pp(self.delta)} {move}.")


def estimate_shift(factory: ModelFactory, data: pd.DataFrame, schema: Schema, window_days: float = 14,
                   min_shift: float = 0.7, min_z: float = 2.0, min_train: int = 20) -> DriftShift | None:
    days = run_days(data, schema)
    if days is None:
        return None
    recent = days >= days.max() - window_days
    if recent.sum() < 3 or (~recent).sum() < min_train:
        return None
    y = data[schema.target.name].to_numpy(float)
    m = factory().fit(data[~recent])
    e = y[recent] - m.predict(data[recent]).mean
    delta, se = float(e.mean()), float(e.std(ddof=1) / np.sqrt(len(e)))
    applied = abs(delta) >= min_shift and (se == 0 or abs(delta) / se >= min_z)
    return DriftShift(delta, se, int(recent.sum()), window_days, bool(applied))


@dataclass
class CalibratedModel:
    base: Surrogate
    factor: float
    drift: DriftShift | None = None

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def schema(self) -> Schema:
        return self.base.schema

    @property
    def shift(self) -> float:
        return self.drift.delta if self.drift is not None and self.drift.applied else 0.0

    def fit(self, data: pd.DataFrame) -> "CalibratedModel":
        self.base.fit(data)
        return self

    def predict(self, rows: pd.DataFrame) -> Prediction:
        p = self.base.predict(rows)
        sd = p.sd * self.factor
        if self.shift:
            sd = np.sqrt(sd**2 + self.drift.se**2)
        return Prediction(p.mean + self.shift, sd, p.novel)


def fit_calibrated(factory: ModelFactory, data: pd.DataFrame, schema: Schema, coverage: float = 0.8,
                   drift_correction: bool = True):
    """Fit on all data with ranges widened by time-ordered errors. Returns (model, errors).

    With ``drift_correction``, a clear shift in the last two weeks is moved out of the
    recent yields before fitting and added to every prediction (see the module notes)."""
    errors = rolling_origin_errors(factory, data, schema)
    drift = estimate_shift(factory, data, schema) if drift_correction else None
    fit_on = data
    if drift is not None and drift.applied:
        days = run_days(data, schema)
        fit_on = data.copy()
        y = schema.target.name
        recent = days >= days.max() - drift.window_days
        fit_on[y] = fit_on[y].astype(float) - np.where(recent, drift.delta, 0.0)
    model = CalibratedModel(factory().fit(fit_on), widen_factor(errors, coverage), drift)
    return model, errors
