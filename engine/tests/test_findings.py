import numpy as np

from yieldengine.explain import unsupported_numbers

from yieldengine.analysis import (STRONG, WEAK, Judgment, discover, find_groups, meeting_agenda,
                                  partial_dependence, strength_label)


def test_strength_label():
    assert strength_label(True, True) == STRONG
    assert strength_label(False, False) == WEAK


def test_groups_alias_pressure_and_gas(history, schema):
    groups = find_groups(history, schema)
    assert any(set(g.members) == {"pressure", "gas_flow"} for g in groups)
    assert any(g.members == ["catalyst"] for g in groups)


def test_pd_hides_no_data(fitted, history, schema):
    model, _ = fitted
    pdf = partial_dependence(model, history, schema, "temp", at={"catalyst": "B"})
    assert not pdf.loc[pdf.value >= 192, "supported"].any()
    assert pdf.loc[(pdf.value > 165) & (pdf.value < 182), "supported"].all()


def test_discover_finds_planted_structure(history, schema, factory, fitted):
    findings = discover(history, schema, factory, errors=fitted[1])
    by_key = {f.key: f for f in findings}
    assert by_key["interaction:temp|catalyst"].facts["category"] == "B"
    assert by_key["interaction:temp|catalyst"].show
    assert "aliasing:gas_flow+pressure" in by_key
    assert "no_effect:stir" in by_key
    assert "drift:recent" in by_key and by_key["drift:recent"].facts["reference_delta_pp"] < 0
    # every number in a sentence is in its facts: the LLM guard checks rephrasings against facts
    for f in findings:
        assert f.strength in ("근거 강함", "보통", "약함")
        assert unsupported_numbers(f.sentence, f.facts) == [], f.sentence

    agenda = meeting_agenda(findings, {"no_effect:stir": Judgment.ODD, "aliasing:gas_flow+pressure": "맞아요"})
    assert [f.key for f in agenda] == ["no_effect:stir"]
