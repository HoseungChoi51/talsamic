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
    """Google TabFM, PyTorch build. Takes a DataFrame and encodes categoricals itself.

    The tabfm package (1.0.1, on PyPI and GitHub main) only ships the v1.0.0 loader, so
    for ``version="1.1.0"`` the 1.1.0 weights (google/tabfm-1.1.0-pytorch) are
    downloaded and loaded through it. If that fails, the 1.0.0 weights are used and
    ``self.version`` says so."""

    name = "tabfm"
    license = "TabFM Non-Commercial License v1.0 (weights); code Apache-2.0"
    commercial_use = False
    install = 'pip install "tabfm[pytorch]" huggingface_hub'
    load_args = ("ckpt", "device", "version")
    REPOS = {"1.1.0": "google/tabfm-1.1.0-pytorch", "1.0.0": "google/tabfm-1.0.0-pytorch"}

    def __init__(self, schema, *args, n_estimators: int = 8, **kw):
        self.n_estimators = n_estimators
        super().__init__(schema, *args, **kw)
        self.version = self._runner[2]

    def load(self, ckpt=None, device="auto", version="1.1.0"):
        tabfm = require("tabfm", self.install)
        loader = getattr(tabfm, "tabfm_v1_1_0_pytorch", None) or getattr(tabfm, "tabfm_v1_0_0_pytorch", None)
        if loader is None:
            raise ModelUnavailable("tabfm is installed without the PyTorch build; " + self.install)
        dev = resolve_device(device)
        native = "1.1.0" if hasattr(tabfm, "tabfm_v1_1_0_pytorch") else "1.0.0"
        if ckpt is None and version != native:
            try:
                from huggingface_hub import snapshot_download

                local = snapshot_download(self.REPOS[version], allow_patterns=["regression/*", "regression/**"])
                return (tabfm.TabFMRegressor,
                        loader.load(model_type="regression", checkpoint_path=local, device=dev), version)
            except Exception as e:
                import warnings

                warnings.warn(f"TabFM {version} weights did not load with the {native} loader ({e}); using {native}")
        try:
            model = loader.load(model_type="regression", checkpoint_path=ckpt, device=dev)
        except Exception as e:
            raise ModelUnavailable(f"TabFM weights could not be loaded: {e}") from e
        return (tabfm.TabFMRegressor, model, "local" if ckpt else native)

    def _call(self, Xc, yc, Xq):
        Regressor, model, _ = self._runner
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
    """Xiaomi-TabLDM, sklearn-style regressor, per its model card.

    Not usable as of 2026-11-02: the Hugging Face repo (occams/Xiaomi-TabLDM, a personal
    account) contains only a README and pickled .ckpt weights. There's no ``tabldm`` package
    and no loader code to review. Unpickling unreviewed checkpoints can run arbitrary code,
    so the weights are not loaded. The adapter is kept for when official code is published."""

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
