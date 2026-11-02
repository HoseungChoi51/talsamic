import numpy as np
import pandas as pd
import pytest

from yieldengine.cleaning import ColumnMapping, convert_units, flag_suspect_values
from yieldengine.explain import checked_reason, unsupported_numbers
from yieldengine.korean import josa
from yieldengine.models import GPSurrogate, LimiXSurrogate, model_factory
from yieldengine.schema import Role


def test_schema_roles(schema):
    names = [v.name for v in schema.features]
    assert "peak_temp" not in names and "yield" not in names  # measured/target never model inputs
    assert {"lot", "days"} <= set(names)
    s2 = schema.with_roles({"stir": "ignore"})
    assert "stir" not in [v.name for v in s2.controls]
    assert schema["stir"].role == Role.SET_BEFORE


def test_simulator_plants_structure(history):
    assert len(history) == 140
    assert history[["pressure", "gas_flow"]].corr(method="spearman").iloc[0, 1] > 0.9
    assert history["is_reference"].sum() == 28
    assert history.loc[history.catalyst == "B", "temp"].max() <= 186


def test_josa():
    assert josa("압력", "과") == "압력과"
    assert josa("온도", "은") == "온도는"
    assert josa("반응 시간", "을") == "반응 시간을"
    assert josa("원료 lot", "이") == "원료 lot이"


def test_factory_falls_back_to_gp(schema):
    with pytest.warns(UserWarning):
        f = model_factory(schema)  # LimiX code isn't installed in CI
    assert f.name == "gp" and f.fallback_reason
    assert isinstance(f(), GPSurrogate)


class FakeLimiX:
    """Stands in for LimiXPredictor: nearest-neighbour mean in coded space."""

    def predict(self, Xc, yc, Xq, task_type):
        assert task_type == "Regression"
        scale = Xc.std(axis=0) + 1e-9
        d = (((Xq[:, None, :] - Xc[None, :, :]) / scale) ** 2).sum(-1)
        idx = np.argsort(d, axis=1)[:, :5]
        return yc[idx].mean(axis=1)


def test_limix_adapter_contract(schema, history):
    m = LimiXSurrogate(schema, predictor=FakeLimiX(), n_bootstrap=4).fit(history)
    p = m.predict(history.head(7))
    assert p.mean.shape == (7,) and np.all(p.sd > 0)
    lo, hi = p.interval(0.8)
    assert np.all(lo < hi)


def test_calibration_only_widens(fitted):
    model, errors = fitted
    assert model.factor >= 1.0
    assert {"actual", "mean", "sd", "z"} <= set(errors.columns)


def test_llm_number_guard():
    facts = {"predicted": 81.3, "low": 78.8, "high": 83.9}
    template = "예상 수율 81.3% (범위 78.8–83.9%)."
    ok = "이 조건은 수율 81.3%가 예상되고 78.8–83.9% 사이일 가능성이 높습니다."
    bad = "이 조건은 수율 85%가 예상됩니다."
    assert checked_reason(ok, facts, template) == (ok, True)
    assert checked_reason(bad, facts, template) == (template, False)
    assert unsupported_numbers(bad, facts, template) == [85.0]


def test_unit_conversion_and_mapping(schema, tmp_path):
    assert convert_units(pd.Series([443.15]), "K", "℃").iloc[0] == pytest.approx(170)
    assert list(convert_units(pd.Series(["1.5h", "90분", "120"]), "text", "분")) == [90, 90, 120]
    raw = pd.DataFrame({"반응T": [443.15], "시간": ["2h"], "cat": [" B "]})
    m = ColumnMapping({"반응T": {"to": "temp", "unit": "K"}, "시간": {"to": "time", "unit": "text"},
                       "cat": {"to": "catalyst"}}, confirmed_by="engineer")
    m.save(tmp_path / "map.json")
    out = ColumnMapping.load(tmp_path / "map.json").apply(raw, schema)
    assert out.loc[0, "temp"] == pytest.approx(170) and out.loc[0, "time"] == 120 and out.loc[0, "catalyst"] == "B"


def test_typo_flagged(history, schema):
    bad = history.copy()
    bad.loc[10, "pressure"] = 15.2  # 1.52 typed wrong
    bad.loc[20, "gas_flow"] = 12.0  # possible, but inconsistent with its pressure
    flags = flag_suspect_values(bad, schema)
    hits = set(zip(flags["row"], flags["column"]))
    assert (10, "pressure") in hits
    assert (20, "gas_flow") in hits
