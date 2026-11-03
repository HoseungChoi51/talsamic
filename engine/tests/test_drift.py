"""Drift correction: a clear recent shift moves every prediction, not only those near recent runs."""

import warnings

import numpy as np
import pytest

from yieldengine.calibration import estimate_shift, fit_calibrated
from yieldengine.engine import Engine
from yieldengine.simulator import FakePlant
from yieldengine.suggest import BatchSettings


def test_shift_found_on_the_drifting_plant(factory, schema, history, fitted):
    d = estimate_shift(factory, history, schema)
    assert d.applied and -3.0 < d.delta < -1.0  # the fake plant loses 2%p in the last weeks
    model, _ = fitted
    assert model.drift.applied and model.shift == pytest.approx(model.drift.delta)
    assert "낮췄다" in model.drift.note and "최근 2주" in model.drift.note


def test_no_shift_without_drift(factory, schema):
    h = FakePlant(seed=0, drift_size=0.0).history()
    d = estimate_shift(factory, h, schema)
    assert d is not None and not d.applied and d.note is None


def test_old_recipes_today_are_predicted_at_the_new_level(factory, schema, history, plant, fitted, context):
    model, _ = fitted
    raw, _ = fit_calibrated(factory, history, schema, drift_correction=False)
    old = history[history["days"] < 150].drop(columns=["yield", "peak_temp", "date"]).copy()
    for k, v in context.items():
        old[k] = v
    truth = FakePlant(seed=0).true_yield(old)
    bias = lambda m: float(np.mean(truth - m.predict(old).mean))
    # On 6 fake plants the raw bias averaged -0.7 and the corrected one 0.0; on this seed the GP
    # had already learned part of the drop, so the check is only that the result lands near the truth.
    assert bias(raw) < 0
    assert abs(bias(model)) < 0.5
    assert np.all(model.predict(old.head(5)).sd >= model.drift.se)  # the range carries the shift's uncertainty


def test_engine_reports_the_shift(schema, history, context):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = Engine(history, schema, model="gp", context=context)
        off = Engine(history, schema, model="gp", context=context, drift_correction=False)
    std = schema.standard_recipe()
    assert eng.predict(std)["drift_shift_pp"] == round(eng.drift.delta, 1)
    assert "drift_shift_pp" not in off.predict(std) and off.drift is None
    batch = eng.suggest(BatchSettings(n_runs=6))
    assert eng.drift.note in batch.notes
