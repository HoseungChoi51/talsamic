import json

import numpy as np
import pytest

from yieldengine.benchmarks.datasets import _schema
from yieldengine.llm.client import parse_json
from yieldengine.llm.descriptors import attach, load_tables, propose, save
from yieldengine.models import model_factory
from yieldengine.schema import FeatureEncoder
from tests.test_benchmarks import _toy


class FakeLLM:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, system, user, **kw):
        assert "steric_bulk" in user and "L1" in user
        return self.replies.pop(0)


def _reply(bulk_l1=1, extra=None):
    vals = {f"L{i}": {"steric_bulk": v, "electron_richness": 6 - v, "cone_angle_deg": 120 + 20 * v,
                      "biaryl_phosphine": int(v > 2), "bidentate": 0}
            for i, v in zip(range(1, 5), (bulk_l1, 2, 5, 3))}
    vals.update(extra or {})
    return "<think>hmm</think>```json\n" + json.dumps({"values": vals, "notes": "ok"}) + "\n```"


def test_parse_json_strips_thinking():
    assert parse_json('<think>{"x": 1}</think> {"y": 2}') == {"y": 2}


def test_propose_validates_and_takes_median():
    b = _toy()
    var = b.schema["ligand"]
    bad = _reply(bulk_l1=9)  # out of range -> dropped
    t = propose(var, FakeLLM([_reply(1), _reply(2), bad]), n_samples=3)
    assert t.values["steric_bulk"]["L1"] == 1.5  # median of 1, 2 (9 was dropped)
    assert t.spread["steric_bulk"]["L1"] == 1.0
    assert any("outside" in i for i in t.issues)
    assert t.confirmed_by is None


def test_attach_adds_model_inputs(tmp_path):
    b = _toy()
    t = propose(b.schema["ligand"], FakeLLM([_reply()]), n_samples=1)
    save([t], tmp_path / "toy.json")
    schema = attach(b.schema, load_tables(tmp_path / "toy.json"))
    enc = FeatureEncoder(schema)
    assert "ligand.steric_bulk" in enc.numeric_names() and "ligand.bidentate" not in enc.numeric_names()
    X = enc.numeric(b.data)
    assert X.shape[1] == len(enc.numeric_names()) and np.all((X >= 0) & (X <= 1))
    coded, cat_idx = enc.coded(b.data)
    assert coded.shape[1] == len(enc.coded_names())
    with pytest.raises(ValueError):
        attach(b.schema, [t], require_confirmed=True)


def test_descriptors_help_on_unseen_category():
    """With descriptors, the GP can predict a ligand it never saw from similar ones."""
    b = _toy()
    bulk = {"L1": 1, "L2": 2, "L3": 5, "L4": 3}  # the toy chemistry: yield rises with ligand bulk
    b.data["yield"] = 12 * b.data["ligand"].map(bulk) + 5 * (b.data["base"] == "B2")
    t = propose(b.schema["ligand"], FakeLLM([_reply()]), n_samples=1)
    train, test = b.data[b.data.ligand != "L4"], b.data[b.data.ligand == "L4"]
    err = {}
    for name, schema in (("plain", b.schema), ("llm", attach(b.schema, [t]))):
        m = model_factory(schema, "gp", n_restarts=0)().fit(train)
        err[name] = float(np.sqrt(np.mean((m.predict(test).mean - test["yield"].to_numpy()) ** 2)))
    assert err["llm"] < 0.7 * err["plain"]


class FlakyLLM(FakeLLM):
    """None in the reply list = this call times out."""

    def chat(self, system, user, **kw):
        r = self.replies.pop(0)
        if r is None:
            raise TimeoutError("timed out")
        return r


def test_one_usable_sample_has_no_spread():
    t = propose(_toy().schema["ligand"], FlakyLLM([None, _reply(), None]), n_samples=3)
    assert t.n_samples == 1 and t.spread["steric_bulk"]["L1"] is None  # unknown, not "confident"
    assert sum("timed out" in i for i in t.issues) == 2


def test_cli_redoes_only_named_variables(tmp_path, monkeypatch):
    import yieldengine.benchmarks.datasets as datasets
    import yieldengine.llm.descriptors as descriptors

    base_reply = json.dumps({"values": {b: {"pka_conjugate_acid": 10, "strength": s, "organic": 0, "cation_size": 3,
                                            "steric_bulk": 1} for b, s in (("B1", 1), ("B2", 4))}})

    class ByVariable:
        model = "fake"

        def __init__(self, **kw):
            self.kw = kw

        def chat(self, system, user, **kw):
            return _reply(bulk_l1=bulk[0]) if "L1" in user else base_reply

    monkeypatch.setattr(datasets, "load", lambda name, descriptors=None: _toy())
    monkeypatch.setattr(descriptors, "ChatClient", ByVariable)
    bulk = [1]
    descriptors.main(["--benchmark", "direct_arylation", "--out", str(tmp_path), "--samples", "1"])
    first = {t.variable: t for t in load_tables(tmp_path / "direct_arylation.json")}
    bulk[0] = 4
    descriptors.main(["--benchmark", "direct_arylation", "--out", str(tmp_path), "--samples", "1",
                      "--variables", "ligand", "--timeout", "60"])
    second = {t.variable: t for t in load_tables(tmp_path / "direct_arylation.json")}
    assert list(second) == ["ligand", "base"]
    assert first["ligand"].values["steric_bulk"]["L1"] == 1 and second["ligand"].values["steric_bulk"]["L1"] == 4
    assert second["base"].created_at == first["base"].created_at  # kept as it was


def test_category_names_with_stray_spaces_still_match():
    b = _toy()
    var = b.schema["ligand"]
    spaced = type(var)(**{**var.__dict__, "categories": ["L1 ", "L2", "L3", "L4"]})
    t = propose(spaced, FakeLLM([_reply()]), n_samples=1)
    assert t.values["steric_bulk"]["L1 "] == 1 and not t.issues
    schema = attach(b.schema, [t])  # table says "L1 ", data says "L1"
    assert "L1" in schema["ligand"].descriptors["steric_bulk"]
