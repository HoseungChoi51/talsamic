"""Adapters for the modern tabular foundation models (regression).

Each one is written against the model's published API (model card / package source,
checked 2026-11-02). Weights download from Hugging Face on first use unless ``ckpt``
points at a local file. A missing package or weights raises ``ModelUnavailable`` so
``model_factory`` can fall back to the GP.

Skipped on purpose:
* Kumo Tabular (general-forecast/kumo-tabular-large-classifier): classification only,
  and yield is a continuous target.
* Relational Transformer (stanford-star): predicts over multi-table relational
  databases; one table of runs doesn't need it.

Licenses matter for a tool that will run at a plant:
EXAONE-Tabular and TabFM weights are non-commercial. Mitra and Xiaomi-TabLDM are
Apache-2.0. LimiX-16M has its own Apache-based license.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd

from .base import ModelUnavailable
from .incontext import InContextSurrogate, require, resolve_device


class EXAONETabularSurrogate(InContextSurrogate):
    """LG AI Research EXAONE-Tabular (~21M params). Regressor head: 999 quantiles,
    reduced to a trimmed mean by ``predict``. Numeric arrays only, so categoricals go in
    as integer codes (the coded encoding already does that)."""

    name = "exaone-tabular"
    license = "EXAONE AI Model License 1.2 - NC (weights); code BSD-3-Clause-LG"
    commercial_use = False
    install = 'pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"'
    load_args = ("ckpt", "device", "ensemble_count", "compute_dtype")

    def load(self, ckpt=None, device="auto", ensemble_count=None, compute_dtype=None):
        mod = require("exaonetabular", self.install)
        kw = {"device": resolve_device(device), "seed": self.seed}
        if ckpt:
            kw["weights"] = ckpt
        if ensemble_count:
            kw["ensemble_count"] = ensemble_count
        if compute_dtype:
            kw["compute_dtype"] = compute_dtype
        elif kw["device"] == "cpu":
            kw["compute_dtype"] = "float32"  # float16 is GPU-only
        try:
            return mod.EXAONETabularRegressor.from_pretrained(**kw)
        except Exception as e:
            raise ModelUnavailable(f"EXAONE-Tabular weights could not be loaded: {e}") from e

    def _call(self, Xc, yc, Xq):
        self._runner.fit(Xc, yc)
        return self._runner.predict(Xq)


class TabFMSurrogate(InContextSurrogate):
    """Google TabFM (v1.1.0 if the installed package has it, else v1.0.0), PyTorch build.
    Takes a DataFrame and encodes categoricals itself."""

    name = "tabfm"
    license = "TabFM Non-Commercial License v1.0 (weights); code Apache-2.0"
    commercial_use = False
    install = 'pip install "tabfm[pytorch]"'
    load_args = ("ckpt", "device")

    def __init__(self, schema, *args, n_estimators: int = 8, **kw):
        self.n_estimators = n_estimators
        super().__init__(schema, *args, **kw)

    def load(self, ckpt=None, device="auto"):
        tabfm = require("tabfm", self.install)
        loader = getattr(tabfm, "tabfm_v1_1_0_pytorch", None) or getattr(tabfm, "tabfm_v1_0_0_pytorch", None)
        if loader is None:
            raise ModelUnavailable("tabfm is installed without the PyTorch build; " + self.install)
        try:
            model = loader.load(model_type="regression", checkpoint_path=ckpt, device=resolve_device(device))
        except Exception as e:
            raise ModelUnavailable(f"TabFM weights could not be loaded: {e}") from e
        return (tabfm.TabFMRegressor, model)

    def _call(self, Xc, yc, Xq):
        Regressor, model = self._runner
        reg = Regressor(model=model, n_estimators=self.n_estimators, random_state=self.seed)
        reg.fit(self.frame(Xc), yc)
        return reg.predict(self.frame(Xq))


class MitraSurrogate(InContextSurrogate):
    """AutoGluon Mitra, used zero-shot (no fine-tuning) through TabularPredictor.
    The mitra-finetune package (1 h bagged fine-tune per fit) is too slow for the many
    refits the 발견 analyses do; it can be added later as a final-model option."""

    name = "mitra"
    license = "Apache-2.0"
    commercial_use = True
    install = 'pip install "autogluon.tabular[mitra]>=1.6"'
    load_args = ("device",)

    def load(self, device="auto"):
        ag = require("autogluon.tabular", self.install)
        return (ag.TabularPredictor, resolve_device(device))

    def _call(self, Xc, yc, Xq):
        Predictor, device = self._runner
        train = self.frame(Xc).assign(__y__=yc)
        with tempfile.TemporaryDirectory() as tmp:
            p = Predictor(label="__y__", problem_type="regression", path=tmp, verbosity=0)
            p.fit(train, hyperparameters={"MITRA": {"fine_tune": False}},
                  num_gpus=1 if device.startswith("cuda") else 0)
            return np.asarray(p.predict(self.frame(Xq)), float)


class TabLDMSurrogate(InContextSurrogate):
    """Xiaomi-TabLDM (occams/Xiaomi-TabLDM). sklearn-style regressor."""

    name = "tabldm"
    license = "Apache-2.0"
    commercial_use = True
    install = "clone https://huggingface.co/occams/Xiaomi-TabLDM and `pip install .` inside it"
    load_args = ("ckpt", "device")

    def load(self, ckpt=None, device="auto"):
        mod = require("tabldm", self.install)
        kw = {"device": None if device == "auto" else device}
        if ckpt:
            kw["model_path"] = ckpt
        try:
            return mod.TabLDMRegressor(**kw)
        except Exception as e:
            raise ModelUnavailable(f"Xiaomi-TabLDM could not be loaded: {e}") from e

    def _call(self, Xc, yc, Xq):
        self._runner.fit(Xc, yc)
        return self._runner.predict(Xq)
