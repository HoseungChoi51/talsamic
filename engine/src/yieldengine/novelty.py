"""New-chemical guard: recipes that use a category the past runs never used.

A tabular model learns what a ligand, base, catalyst or additive does only from runs that
used it. For a recipe with a new one, the prediction is a guess. The benchmarks showed the
in-context models are overconfident there: with held-out ligands on direct arylation,
their 80% ranges covered as little as 13% of the actual yields.

The guard wraps any model:

* each predicted row lists the conditions whose value has fewer than ``min_runs`` past
  runs (``Prediction.novel``), so the UI can say the prediction is a guess;
* those rows' ranges are widened to at least the error measured by holding out one value
  at a time from the past runs and predicting its runs from the rest (leave-one-level-out).
  The range then shows how wrong the model has been on a new value, not how sure it feels;
* the predicted mean is left alone. Descriptors (``llm/descriptors.py``) are what can
  improve it, and the held-out error measured here gets smaller when they help.

Only set-before conditions are checked. A new raw-material lot (a context column) is the
drift analysis's job. Like calibration, the guard only widens ranges, never narrows them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .models import ModelFactory, Prediction, Surrogate
from .schema import Schema


class NoveltyGuard:
    def __init__(self, factory: ModelFactory, schema: Schema, min_runs: int = 1, max_levels: int = 6,
                 max_rows_per_level: int = 20, min_train: int = 5, seed: int = 0):
        self.factory = factory
        self.schema = schema
        self.min_runs = min_runs
        self.max_levels = max_levels
        self.max_rows_per_level = max_rows_per_level
        self.min_train = min_train
        self.seed = seed
        self.variables = [v for v in schema.controls if not v.is_continuous]
        self.base: Surrogate | None = None
        self._level_error: dict[str, float] = {}

    @property
    def name(self) -> str:
        return getattr(self.base, "name", None) or self.factory.name

    def fit(self, data: pd.DataFrame) -> "NoveltyGuard":
        self.data = data.reset_index(drop=True)
        self.base = self.factory().fit(self.data)
        self.counts = {v.name: self.data[v.name].astype(str).value_counts() for v in self.variables}
        self._level_error = {}
        return self

    # -- which rows are new -------------------------------------------------------------
    def new_conditions(self, rows: pd.DataFrame) -> list[tuple[str, ...]]:
        """For each row, the conditions whose value has fewer than ``min_runs`` past runs."""
        return _flags(self.variables, self.counts, rows, self.min_runs)

    # -- how wrong the model is on a new value --------------------------------------------
    def _point(self, train: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
        m = self.factory()
        fast = getattr(m, "point_predict", None)  # in-context models: one forward pass
        return fast(train, rows) if fast else m.fit(train).predict(rows).mean

    def level_error(self, name: str) -> float:
        """RMSE when one value of ``name`` is held out and its runs are predicted from the rest.
        With fewer than two values tried so far there's nothing to hold out, so the spread of
        all past yields is used instead (the model knows nothing about the new value)."""
        if name in self._level_error:
            return self._level_error[name]
        y_name = self.schema.target.name
        levels = [lv for lv, c in self.counts[name].items() if c >= 1]
        rng = np.random.default_rng(self.seed)
        if len(levels) > self.max_levels:
            levels = list(rng.choice(levels, self.max_levels, replace=False))
        col = self.data[name].astype(str)
        errs = []
        for lv in levels if len(levels) >= 2 else []:
            held, train = self.data[col == lv], self.data[col != lv]
            if len(train) < self.min_train:
                continue
            if len(held) > self.max_rows_per_level:
                held = held.sample(self.max_rows_per_level, random_state=self.seed)
            errs.append(held[y_name].to_numpy(float) - self._point(train, held))
        if errs:
            err = float(np.sqrt(np.mean(np.concatenate(errs) ** 2)))
        else:
            err = float(np.std(self.data[y_name].to_numpy(float)))
        self._level_error[name] = err
        return err

    def predict(self, rows: pd.DataFrame) -> Prediction:
        assert self.base is not None, "call fit first"
        p = self.base.predict(rows)
        novel = self.new_conditions(rows)
        sd = np.asarray(p.sd, float).copy()
        for i, names in enumerate(novel):
            if names:  # several new values at once: the largest single-value error, a lower bound
                sd[i] = max(sd[i], max(self.level_error(n) for n in names))
        return Prediction(p.mean, sd, novel)


def _flags(variables, counts, rows: pd.DataFrame, min_runs: int) -> list[tuple[str, ...]]:
    names, cols = [], []
    for v in variables:
        if v.name in rows:
            n = rows[v.name].astype(str).map(counts[v.name]).fillna(0).to_numpy()
            names.append(v.name)
            cols.append(n < min_runs)
    if not cols:
        return [()] * len(rows)
    return [tuple(n for n, f in zip(names, row) if f) for row in zip(*cols)]


def new_conditions(schema: Schema, past: pd.DataFrame, rows: pd.DataFrame, min_runs: int = 1):
    """Without a model: for each row, the set-before conditions whose value ``past`` used
    fewer than ``min_runs`` times."""
    variables = [v for v in schema.controls if not v.is_continuous]
    counts = {v.name: past[v.name].astype(str).value_counts() for v in variables}
    return _flags(variables, counts, rows, min_runs)


def guarded(factory: ModelFactory, schema: Schema, **kw) -> ModelFactory:
    """A factory whose models flag new categorical values and widen their ranges."""
    return ModelFactory(lambda: NoveltyGuard(factory, schema, **kw), factory.name, factory.fallback_reason)
