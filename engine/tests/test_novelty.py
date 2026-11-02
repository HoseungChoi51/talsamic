"""The new-chemical guard: flag values the past runs never used, widen only those ranges."""

import numpy as np
import pytest

from yieldengine.models import InContextSurrogate, ModelFactory
from yieldengine.novelty import NoveltyGuard, guarded
from yieldengine.suggest import BatchSettings, suggest_batch
from yieldengine.whatif import predict_recipe


@pytest.fixture(scope="module")
def no_c(history):
    return history[history["catalyst"].astype(str) != "C"].reset_index(drop=True)


@pytest.fixture(scope="module")
def guard(factory, schema, no_c):
    return NoveltyGuard(factory, schema).fit(no_c)


def test_flags_only_unseen_values(guard, history):
    rows = history.iloc[:30]
    p = guard.predict(rows)
    is_c = rows["catalyst"].astype(str).eq("C").to_numpy()
    assert is_c.any() and (~is_c).any()
    assert [bool(n) for n in p.novel] == list(is_c)
    assert all(n == ("catalyst",) for n, c in zip(p.novel, is_c) if c)


def test_widens_new_rows_and_leaves_the_rest(guard, history):
    rows = history.iloc[:30]
    base = guard.base.predict(rows)
    p = guard.predict(rows)
    is_c = rows["catalyst"].astype(str).eq("C").to_numpy()
    np.testing.assert_allclose(p.mean, base.mean)
    np.testing.assert_allclose(p.sd[~is_c], base.sd[~is_c])
    assert np.all(p.sd[is_c] >= guard.level_error("catalyst") - 1e-9)
    assert np.all(p.sd >= base.sd - 1e-9)  # never narrower


def test_one_value_tried_falls_back_to_the_yield_spread(factory, schema, history):
    only_a = history[history["catalyst"].astype(str) == "A"].reset_index(drop=True)
    g = NoveltyGuard(factory, schema).fit(only_a)
    assert g.level_error("catalyst") == pytest.approx(only_a["yield"].std(ddof=0))


def test_in_context_models_use_one_pass(schema, no_c, history):
    calls = {"pass": 0}

    class Mean(InContextSurrogate):
        name = "mean"

        def load(self, **kw):
            return object()

        def _call(self, Xc, yc, Xq):
            calls["pass"] += 1
            return np.full(len(Xq), float(np.mean(yc)))

    f = ModelFactory(lambda: Mean(schema, n_bootstrap=2, n_folds=0), "mean")
    g = NoveltyGuard(f, schema).fit(no_c)
    before = calls["pass"]
    g.level_error("catalyst")  # two values tried (A, B): one pass each
    assert calls["pass"] - before == 2


def test_engine_paths_say_it_is_a_guess(factory, schema, no_c, history, context):
    model = guarded(factory, schema)().fit(no_c)
    recipe = {**schema.standard_recipe(), "catalyst": "C"}
    out = predict_recipe(model, recipe, context)
    assert out["new_conditions"] == ["catalyst"]
    assert "new_conditions" not in predict_recipe(model, schema.standard_recipe(), context)

    pool = history[[v.name for v in schema.controls]].drop_duplicates().head(200)
    b = suggest_batch(model, no_c, schema, BatchSettings(n_runs=8, split=(6, 2, 0), risk="도전적으로",
                                                        max_changes=len(schema.controls)), context,
                      candidates=pool)
    new = [s for s in b.suggestions if s.recipe["catalyst"] == "C"]
    assert new, "a wide range on a new catalyst should attract 도전적으로 or learn runs"
    for s in new:
        assert s.facts["new_conditions"] == ["catalyst"] and "처음 써 보는 조건(촉매)" in s.reason
        note = s.reason.split("처음 써 보는")[1]
        assert not any(ch.isdigit() for ch in note)  # adds no number for the LLM guard to check
    for s in b.suggestions:
        if s.recipe["catalyst"] != "C":
            assert "처음 써 보는" not in s.reason


def _opt_with_c(model, no_c, schema, context, history, **kw):
    pool = history[[v.name for v in schema.controls]].drop_duplicates()
    b = suggest_batch(model, no_c, schema, BatchSettings(n_runs=8, split=(6, 2, 0),
                                                        max_changes=len(schema.controls), **kw), context,
                      candidates=pool)
    return b, sum(s.recipe["catalyst"] == "C" for s in b.suggestions if s.purpose == "최적화")


@pytest.mark.parametrize("risk, limit", [("안전하게", 0), ("균형", 1)])
def test_risk_level_limits_guesses_in_optimize_runs(factory, schema, no_c, history, context, risk, limit):
    model = guarded(factory, schema)().fit(no_c)
    _, n = _opt_with_c(model, no_c, schema, context, history, risk=risk)
    assert n <= limit
    _, n2 = _opt_with_c(model, no_c, schema, context, history, risk=risk, max_new=6)
    assert n2 >= n


def test_fixed_new_value_does_not_count(factory, schema, no_c, history, context):
    model = guarded(factory, schema)().fit(no_c)
    b, n = _opt_with_c(model, no_c, schema, context, history, risk="안전하게", fixed={"catalyst": "C"})
    assert n == 6 and not any("처음 써 보는" in note for note in b.notes)
