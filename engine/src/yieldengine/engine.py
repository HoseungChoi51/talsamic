"""One object the UI and the LLM tools both call. Each screen maps to one method.

    이번 주   -> Engine.this_week(saved_batch, actuals)
    데이터 정리 -> Engine.suspect_values()  (+ cleaning.ColumnMapping for uploads)
    발견      -> Engine.findings()
    바꿔보기   -> Engine.predict(recipe), Engine.slider_curves(recipe)
    다음 실험   -> Engine.suggest(BatchSettings(...))

LLM tools should get the read-only methods plus a "propose settings" tool that returns a
``BatchSettings`` for an engineer to click; never ``suggest`` itself.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any

import pandas as pd

from .analysis import Finding, discover
from .calibration import fit_calibrated
from .cleaning import flag_suspect_values
from .models import DEFAULT_MODEL, license_note, model_factory
from .novelty import guarded
from .schema import Schema
from .suggest import Batch, BatchSettings, suggest_batch
from .tracking import compare_with_actuals
from .whatif import predict_recipe, slider_curves


class Engine:
    def __init__(self, data: pd.DataFrame, schema: Schema, model: str = DEFAULT_MODEL,
                 context: dict[str, Any] | None = None, **model_kw):
        self.data = data.reset_index(drop=True)
        self.schema = schema
        # every prediction flags recipes with a value the past runs never used (novelty.py)
        self.factory = guarded(model_factory(schema, model, **model_kw), schema)
        self.context = context if context is not None else self._latest_context()

    def _latest_context(self) -> dict[str, Any]:
        ctx = {}
        last = self.data.sort_values("days").iloc[-1] if "days" in self.data else self.data.iloc[-1]
        for v in self.schema.context:
            ctx[v.name] = float(last[v.name]) + 1 if v.name == "days" else last[v.name]
        return ctx

    @property
    def model_name(self) -> str:
        return self.factory.name

    @property
    def fallback_reason(self) -> str | None:
        return self.factory.fallback_reason

    @property
    def license_note(self) -> str | None:
        """Shown next to the model name when its license may not allow plant use (TabFM, EXAONE)."""
        return license_note(self.model_name)

    @cached_property
    def _fitted(self):
        return fit_calibrated(self.factory, self.data, self.schema)

    @property
    def model(self):
        return self._fitted[0]

    @property
    def calibration_errors(self) -> pd.DataFrame:
        return self._fitted[1]

    @cached_property
    def _findings(self) -> list[Finding]:
        return discover(self.data, self.schema, self.factory, errors=self.calibration_errors)

    def findings(self, include_weak: bool = False) -> list[Finding]:
        return [f for f in self._findings if include_weak or f.show]

    def predict(self, recipe: dict[str, Any]) -> dict:
        return predict_recipe(self.model, recipe, self.context)

    def slider_curves(self, recipe: dict[str, Any]) -> dict[str, pd.DataFrame]:
        return slider_curves(self.model, self.data, self.schema, recipe, self.context)

    def suggest(self, settings: BatchSettings | None = None) -> Batch:
        return suggest_batch(self.model, self.data, self.schema, settings, self.context)

    def suspect_values(self) -> pd.DataFrame:
        return flag_suspect_values(self.data, self.schema)

    def this_week(self, saved_batch: pd.DataFrame, actuals: pd.DataFrame):
        return compare_with_actuals(saved_batch, actuals, self.schema.target.name)
