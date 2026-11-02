"""데이터 정리 helpers the engine owns (the LLM only proposes column/unit mappings).

* ``convert_units``: rule-based converters an engineer-approved mapping applies.
* ``ColumnMapping``: saved and reused on the next upload.
* ``flag_suspect_values``: likely typos (15.2 bar instead of 1.52): values outside the
  schema's possible range, or far from what the other columns predict.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict

from .schema import FeatureEncoder, Schema

_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(h|hr|hour|hours|시간|m|min|분)?\s*$", re.I)


def _to_minutes(x) -> float:
    if isinstance(x, (int, float, np.number)):
        return float(x)
    m = _DURATION.match(str(x))
    if not m:
        return np.nan
    val, unit = float(m.group(1)), (m.group(2) or "min").lower()
    return val * 60 if unit in ("h", "hr", "hour", "hours", "시간") else val


CONVERTERS = {
    ("K", "℃"): lambda s: pd.to_numeric(s, errors="coerce") - 273.15,
    ("℉", "℃"): lambda s: (pd.to_numeric(s, errors="coerce") - 32) * 5 / 9,
    ("kPa", "bar"): lambda s: pd.to_numeric(s, errors="coerce") / 100,
    ("MPa", "bar"): lambda s: pd.to_numeric(s, errors="coerce") * 10,
    ("psi", "bar"): lambda s: pd.to_numeric(s, errors="coerce") * 0.0689476,
    ("h", "분"): lambda s: s.map(_to_minutes),
    ("text", "분"): lambda s: s.map(_to_minutes),  # "1.5h", "90분", "90"
}


def convert_units(series: pd.Series, from_unit: str, to_unit: str) -> pd.Series:
    if from_unit == to_unit:
        return pd.to_numeric(series, errors="coerce")
    try:
        return CONVERTERS[(from_unit, to_unit)](series)
    except KeyError:
        raise ValueError(f"no converter from {from_unit} to {to_unit}") from None


@dataclass
class ColumnMapping:
    """Upload column -> schema variable, with unit. Proposed by the LLM, confirmed by an engineer."""

    columns: dict[str, dict] = field(default_factory=dict)  # {"반응T": {"to": "temp", "unit": "K"}}
    confirmed_by: str | None = None

    def apply(self, raw: pd.DataFrame, schema: Schema) -> pd.DataFrame:
        out = pd.DataFrame(index=raw.index)
        for src, spec in self.columns.items():
            var = schema[spec["to"]]
            if var.is_continuous:
                out[var.name] = convert_units(raw[src], spec.get("unit", var.unit.strip()), var.unit.strip())
            else:
                out[var.name] = raw[src].astype(str).str.strip()
        rest = [c for c in raw.columns if c not in self.columns and c not in out.columns]
        return pd.concat([out, raw[rest]], axis=1)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ColumnMapping":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def flag_suspect_values(data: pd.DataFrame, schema: Schema, z: float = 4.0) -> pd.DataFrame:
    """Rows/columns worth a second look, with the reason and the value other columns suggest."""
    flags = []
    cont = [v for v in schema.features if v.is_continuous]
    for v in cont:
        x = data[v.name].astype(float)
        for i in x.index[(x < v.lower) | (x > v.upper)]:
            flags.append({"row": i, "column": v.name, "value": x[i], "expected": None,
                          "reason": f"가능한 범위({v.fmt(v.lower)}–{v.fmt(v.upper)}) 밖"})
    for v in cont:
        others = [w for w in schema.features if w.name != v.name]
        X = FeatureEncoder(schema, others).numeric(data)
        y = data[v.name].astype(float).to_numpy()
        if np.unique(y).size < 5 or len(y) < 30:
            continue
        pred = cross_val_predict(HistGradientBoostingRegressor(max_iter=100, random_state=0), X, y, cv=5)
        resid = y - pred
        mad = np.median(np.abs(resid - np.median(resid))) * 1.4826
        if mad <= 0:
            continue
        for i in np.where(np.abs(resid) > z * mad + 1e-9)[0]:
            flags.append({"row": data.index[i], "column": v.name, "value": y[i], "expected": round(float(pred[i]), 2),
                          "reason": "다른 조건으로 보면 이 값이 어색함"})
    return pd.DataFrame(flags, columns=["row", "column", "value", "expected", "reason"]).drop_duplicates(["row", "column"])
