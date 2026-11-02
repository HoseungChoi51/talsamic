"""Adapters are checked against stand-ins that follow each model's published API."""

import sys
import types

import numpy as np
import pandas as pd
import pytest

from yieldengine.models import (REGISTRY, EXAONETabularSurrogate, LimiXSurrogate, MitraSurrogate, ModelUnavailable,
                                TabFMSurrogate, TabLDMSurrogate, model_factory, model_info)


def _knn(Xc, yc, Xq, k=5):
    Xc, Xq = np.asarray(Xc, float), np.asarray(Xq, float)
    scale = Xc.std(axis=0) + 1e-9
    d = (((Xq[:, None, :] - Xc[None, :, :]) / scale) ** 2).sum(-1)
    return np.asarray(yc)[np.argsort(d, axis=1)[:, :k]].mean(axis=1)


class SklearnStyle:
    """fit(X, y) stores context; predict(X) answers from it."""

    def __init__(self, **kw):
        self.kw = kw

    def fit(self, X, y):
        assert np.asarray(X).dtype.kind == "f", "foundation models get numeric arrays"
        self.X, self.y = np.asarray(X, float), np.asarray(y, float)
        return self

    def predict(self, X):
        return _knn(self.X, self.y, X)


def _fake(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


@pytest.fixture
def split(history):
    return history.iloc[:100], history.iloc[100:]


def _check(model, split):
    train, test = split
    p = model.fit(train).predict(test)
    assert p.mean.shape == (len(test),) and np.all(np.isfinite(p.mean)) and np.all(p.sd > 0)
    rmse = np.sqrt(np.mean((p.mean - test["yield"].to_numpy()) ** 2))
    assert rmse < 2 * test["yield"].std()


def test_limix_adapter(schema, split):
    class FakeLimiX:  # like LimiX v1: answers in standardized units
        def predict(self, Xc, yc, Xq, task_type):
            assert task_type == "Regression" and abs(np.mean(yc)) < 1e-6
            assert len(Xq) >= 2, "real LimiX v1 crashes on one query row"
            return _knn(Xc, yc, Xq)

    m = LimiXSurrogate(schema, predictor=FakeLimiX(), n_bootstrap=3)
    _check(m, split)
    one = m.predict(split[1].head(1))  # a single row (the standard recipe) must work too
    assert one.mean.shape == (1,)


def test_exaone_adapter(schema, split, monkeypatch):
    seen = {}

    class Reg(SklearnStyle):
        @classmethod
        def from_pretrained(cls, **kw):
            seen.update(kw)
            return cls(**kw)

    _fake(monkeypatch, "exaonetabular", EXAONETabularRegressor=Reg)
    m = EXAONETabularSurrogate(schema, device="cpu", n_bootstrap=3)
    assert seen["device"] == "cpu" and seen["compute_dtype"] == "float32"
    _check(m, split)


def test_tabfm_adapter(schema, split, monkeypatch):
    class Reg:
        def __init__(self, model, n_estimators, random_state):
            assert model == "weights"

        def fit(self, X, y):
            assert isinstance(X, pd.DataFrame) and X["catalyst"].dtype == "category"
            self.X, self.y = X.astype(float).to_numpy(), np.asarray(y)
            return self

        def predict(self, X):
            return _knn(self.X, self.y, X.astype(float).to_numpy())

    loader = types.SimpleNamespace(load=lambda model_type, checkpoint_path, device: "weights")
    _fake(monkeypatch, "tabfm", TabFMRegressor=Reg, tabfm_v1_1_0_pytorch=loader)
    m = TabFMSurrogate(schema, n_bootstrap=3)
    assert m.version == "1.1.0"
    _check(m, split)


def test_mitra_adapter(schema, split, monkeypatch):
    class Predictor:
        def __init__(self, label, problem_type, path, verbosity):
            assert problem_type == "regression"
            self.label = label

        def fit(self, df, hyperparameters, num_gpus):
            assert hyperparameters == {"MITRA": {"fine_tune": False}}
            self.X = df.drop(columns=self.label).astype(float).to_numpy()
            self.y = df[self.label].to_numpy()

        def predict(self, df):
            return pd.Series(_knn(self.X, self.y, df.astype(float).to_numpy()))

    _fake(monkeypatch, "autogluon")
    _fake(monkeypatch, "autogluon.tabular", TabularPredictor=Predictor)
    _check(MitraSurrogate(schema, device="cpu", n_bootstrap=2), split)


def test_tabldm_adapter(schema, split, monkeypatch):
    _fake(monkeypatch, "tabldm", TabLDMRegressor=SklearnStyle)
    _check(TabLDMSurrogate(schema, n_bootstrap=3), split)


def test_factory_shares_weights_and_falls_back(schema, monkeypatch):
    loads = []

    class Reg(SklearnStyle):
        @classmethod
        def from_pretrained(cls, **kw):
            loads.append(kw)
            return cls()

    _fake(monkeypatch, "exaonetabular", EXAONETabularRegressor=Reg)
    f = model_factory(schema, "exaone-tabular", device="cpu", n_bootstrap=2)
    a, b = f(), f()
    assert len(loads) == 1 and a._runner is b._runner and a.n_bootstrap == 2

    monkeypatch.delitem(sys.modules, "exaonetabular")
    monkeypatch.setattr("importlib.import_module", lambda n: (_ for _ in ()).throw(ImportError(n)))
    with pytest.warns(UserWarning):
        fb = model_factory(schema, "exaone-tabular")
    assert fb.name == "gp" and "exaone-tabular" in fb.fallback_reason


def test_model_info_flags_noncommercial():
    info = {m["name"]: m for m in model_info()}
    assert set(info) == set(REGISTRY)
    assert info["exaone-tabular"]["commercial_use"] is False and info["tabfm"]["commercial_use"] is False
    assert info["mitra"]["commercial_use"] is True and info["gp"]["commercial_use"] is True


def test_limix_cache_redirect_reaches_nested_class(monkeypatch, tmp_path):
    from yieldengine.models.limix import _redirect_limix_cache

    class LimiXPredictor:
        class CacheManager:
            def __init__(self, cache_dir="/mnt/public/infe_cache"):
                self.cache_dir = cache_dir

    LimiXPredictor.__module__ = LimiXPredictor.CacheManager.__module__ = "inference.v1_0.predictor"
    pkg = types.ModuleType("inference")
    sub = types.ModuleType("inference.v1_0")
    mod = types.ModuleType("inference.v1_0.predictor")
    mod.LimiXPredictor = LimiXPredictor
    sub.predictor = mod
    pkg.v1_0 = sub
    for name, m in (("inference", pkg), ("inference.v1_0", sub), ("inference.v1_0.predictor", mod)):
        monkeypatch.setitem(sys.modules, name, m)
    monkeypatch.setenv("LIMIX_CACHE_DIR", str(tmp_path))
    _redirect_limix_cache()
    assert LimiXPredictor.CacheManager().cache_dir == str(tmp_path)
