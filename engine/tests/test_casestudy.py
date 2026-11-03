"""The case-study runner: planted data problems are what they say they are."""

from yieldengine.casestudy import PROBES, leak_schema, make_typos, truth_optimum
from yieldengine.schema import Role


def test_typos_change_five_values(history, schema):
    bad, log = make_typos(history, schema)
    assert len(log) == 5 and len({(t["row"], t["column"]) for t in log}) == 5
    for t in log:
        assert bad.at[t["row"], t["column"]] == t["typed"] != t["true_value"]
    assert (bad.drop(columns=["date"]) != history.drop(columns=["date"])).to_numpy().sum() == 5


def test_leak_makes_the_measured_column_a_setting(schema):
    s = leak_schema(schema)
    assert s["peak_temp"].role == Role.SET_BEFORE and "peak_temp" in s.standard_recipe()
    assert schema["peak_temp"].role == Role.MEASURED


def test_truth_optimum_uses_catalyst_b(plant, history):
    opt = truth_optimum(plant, plant.context_now(history))
    assert opt["recipe"]["catalyst"] == "B" and 180 <= opt["recipe"]["temp"] <= 190
    assert "standard" in PROBES
