"""Variable definitions and roles.

The role column on 데이터 정리 is the one manual step the engineers own:
only they can say whether a column is set before a run or measured during it.
Everything downstream reads roles from here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

import numpy as np
import pandas as pd


class Role(str, Enum):
    SET_BEFORE = "set_before"  # 실험 전에 정하는 조건 (controllable)
    CONTEXT = "context"  # 원료 lot, 날짜 등: 모델에 넣지만 고르지는 않음
    MEASURED = "measured"  # 실험 중에 측정되는 값: 모델 입력에서 제외 (leakage)
    TARGET = "target"  # 수율
    IGNORE = "ignore"


class Kind(str, Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass
class Variable:
    name: str
    label: str  # Korean label shown in the UI
    kind: Kind
    role: Role
    lower: float | None = None
    upper: float | None = None
    categories: list[str] = field(default_factory=list)
    standard: Any = None  # value in the standard recipe
    unit: str = ""
    step: float | None = None  # resolution engineers can actually set
    # numeric descriptors of each category, e.g. {"steric_bulk": {"XPhos": 4, "PPh3": 2}}.
    # Proposed by the LLM, confirmed by an engineer; they let a model reason about a
    # category it has never seen in the data (see llm/descriptors.py).
    descriptors: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def is_continuous(self) -> bool:
        return self.kind == Kind.CONTINUOUS

    def span(self) -> float:
        assert self.is_continuous
        return float(self.upper - self.lower)

    def fmt(self, value: Any) -> str:
        if self.is_continuous:
            digits = 0 if (self.step or 1) >= 1 else 1 if (self.step or 1) >= 0.1 else 2
            return f"{float(value):.{digits}f}{self.unit}"
        return str(value)


@dataclass
class Schema:
    variables: list[Variable]
    date_column: str | None = "date"
    reference_column: str | None = "is_reference"

    def __post_init__(self) -> None:
        self._by_name = {v.name: v for v in self.variables}
        targets = [v for v in self.variables if v.role == Role.TARGET]
        if len(targets) != 1:
            raise ValueError("schema needs exactly one target variable")

    def __getitem__(self, name: str) -> Variable:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    @property
    def target(self) -> Variable:
        return next(v for v in self.variables if v.role == Role.TARGET)

    @property
    def controls(self) -> list[Variable]:
        return [v for v in self.variables if v.role == Role.SET_BEFORE]

    @property
    def context(self) -> list[Variable]:
        return [v for v in self.variables if v.role == Role.CONTEXT]

    @property
    def features(self) -> list[Variable]:
        """Model inputs: controllable conditions plus context. Never measured columns."""
        return self.controls + self.context

    def standard_recipe(self) -> dict[str, Any]:
        return {v.name: v.standard for v in self.controls}

    def with_roles(self, roles: dict[str, Role | str]) -> "Schema":
        """Return a copy with roles changed (what the 데이터 정리 role column edits)."""
        new = []
        for v in self.variables:
            r = roles.get(v.name)
            new.append(v if r is None else Variable(**{**v.__dict__, "role": Role(r)}))
        return Schema(new, self.date_column, self.reference_column)


class FeatureEncoder:
    """Turns recipe rows into model matrices.

    Two encodings are offered because models differ:
    - ``numeric``: continuous scaled to [0, 1] by the schema bounds, categoricals one-hot.
      Used by the GP and by distance computations.
    - ``coded``: continuous raw, categoricals as integer codes, plus the indices of the
      categorical columns. Used by tabular foundation models that handle scaling and
      categoricals themselves.
    """

    def __init__(self, schema: Schema, variables: Iterable[Variable] | None = None):
        self.schema = schema
        self.variables = list(variables) if variables is not None else schema.features

    def _descriptors(self, df: pd.DataFrame, scaled: bool) -> list[np.ndarray]:
        cols = []
        for v in self.variables:
            for _, table in sorted(v.descriptors.items()):
                vals = np.array(list(table.values()), float)
                lo, hi = np.nanmin(vals), np.nanmax(vals)
                x = df[v.name].astype(str).map(table).astype(float).to_numpy()
                x = np.where(np.isnan(x), np.nanmean(vals), x)  # category with no descriptor
                cols.append((x - lo) / (hi - lo) if scaled and hi > lo else x)
        return cols

    def numeric(self, df: pd.DataFrame) -> np.ndarray:
        cols = []
        for v in self.variables:
            if v.is_continuous:
                cols.append(((df[v.name].astype(float) - v.lower) / v.span()).to_numpy()[:, None])
            else:
                vals = df[v.name].astype(str).to_numpy()
                cols.append(np.stack([(vals == c).astype(float) for c in v.categories], axis=1))
        cols += [c[:, None] for c in self._descriptors(df, scaled=True)]
        return np.hstack(cols) if cols else np.zeros((len(df), 0))

    def descriptor_names(self) -> list[str]:
        return [f"{v.name}.{d}" for v in self.variables for d in sorted(v.descriptors)]

    def numeric_names(self) -> list[str]:
        names = []
        for v in self.variables:
            if v.is_continuous:
                names.append(v.name)
            else:
                names.extend(f"{v.name}={c}" for c in v.categories)
        return names + self.descriptor_names()

    def coded_names(self) -> list[str]:
        return [v.name for v in self.variables] + self.descriptor_names()

    def coded(self, df: pd.DataFrame) -> tuple[np.ndarray, list[int]]:
        cols, cat_idx = [], []
        for i, v in enumerate(self.variables):
            if v.is_continuous:
                cols.append(df[v.name].astype(float).to_numpy())
            else:
                lookup = {c: k for k, c in enumerate(v.categories)}
                cols.append(df[v.name].astype(str).map(lookup).astype(float).to_numpy())
                cat_idx.append(i)
        cols += self._descriptors(df, scaled=False)
        return np.column_stack(cols), cat_idx
