import numpy as np
import pandas as pd
import pytest

from yieldengine.analysis import nearest_distance
from yieldengine.suggest import LEARN, OPTIMIZE, REFERENCE, RISK_LEVELS, BatchSettings, suggest_batch
from yieldengine.tracking import compare_with_actuals, save_batch
from yieldengine.whatif import slider_curves


def _batch(fitted, history, schema, context, **kw):
    return suggest_batch(fitted[0], history, schema, BatchSettings(n_candidates=1500, **kw), context)


def test_default_split_and_reference(fitted, history, schema, context):
    b = _batch(fitted, history, schema, context)
    purposes = [s.purpose for s in b.suggestions]
    assert purposes.count(OPTIMIZE) == 8 and purposes.count(LEARN) == 3 and purposes.count(REFERENCE) == 1
    ref = next(s for s in b.suggestions if s.purpose == REFERENCE)
    assert ref.recipe == schema.standard_recipe() and ref.changes == []
    learn = [s for s in b.suggestions if s.purpose == LEARN]
    assert any(s.facts.get("separates") for s in learn)  # pressure vs gas flow
    for s in b.suggestions:
        assert s.low < s.predicted < s.high


def test_custom_split_must_add_up():
    with pytest.raises(ValueError):
        BatchSettings(n_runs=10, split=(5, 3, 1)).resolved_split()
    assert BatchSettings(n_runs=6).resolved_split() == (4, 1, 1)


def test_constraints_respected(fitted, history, schema, context):
    b = _batch(fitted, history, schema, context, fixed={"catalyst": "B"}, limits={"temp": (160, 185)},
               max_changes=1)
    for s in b.suggestions:
        assert s.recipe["catalyst"] == "B"
        assert 160 <= s.recipe["temp"] <= 185
        non_fixed = [c for c in s.changes if c[0] != "catalyst"]
        assert len(non_fixed) <= 1


@pytest.mark.parametrize("risk", list(RISK_LEVELS))
def test_risk_radius(fitted, history, schema, context, risk):
    b = _batch(fitted, history, schema, context, risk=risk)
    opt = pd.DataFrame([s.recipe for s in b.suggestions if s.purpose == OPTIMIZE])
    d = nearest_distance(schema, opt, history)
    assert np.all(d <= RISK_LEVELS[risk].radius + 1e-9)


def test_risk_changes_ranking(fitted, history, schema, context):
    safe = _batch(fitted, history, schema, context, risk="안전하게")
    bold = _batch(fitted, history, schema, context, risk="도전적으로")
    width = lambda b: np.mean([s.high - s.low for s in b.suggestions if s.purpose == OPTIMIZE])
    assert width(bold) > width(safe)


def test_seed_is_deterministic(fitted, history, schema, context):
    a = _batch(fitted, history, schema, context, seed=3)
    b = _batch(fitted, history, schema, context, seed=3)
    assert [s.recipe for s in a.suggestions] == [s.recipe for s in b.suggestions]


def test_export_and_compare(fitted, history, schema, context, plant, tmp_path):
    b = _batch(fitted, history, schema, context)
    saved = pd.read_csv(save_batch(b, tmp_path))
    ran = plant.run(b.to_frame())
    table, summary = compare_with_actuals(saved, ran[["batch_id", "run", "yield"]])
    assert summary["total"] == 12 and summary["text"].endswith("건 예측 범위 안")
    assert (tmp_path / f"batch_{b.created_at[:10]}_{b.batch_id}.json").exists()


def test_slider_curves(fitted, history, schema, context):
    curves = slider_curves(fitted[0], history, schema, {**schema.standard_recipe(), "catalyst": "B"}, context)
    assert set(curves) == {v.name for v in schema.controls}
    t = curves["temp"]
    assert not t.loc[t.value >= 195, "supported"].any()
    assert t.loc[t.value.between(165, 180), "supported"].all()
    assert len(curves["catalyst"]) == 3
