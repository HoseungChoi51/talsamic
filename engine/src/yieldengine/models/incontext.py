"""Shared adapter for in-context tabular foundation models.

LimiX, EXAONE-Tabular, TabFM, Mitra and Xiaomi-TabLDM all work the same way: the past
runs are the context, the candidates are the queries, and one forward pass predicts
every query. A subclass only has to say how to load the weights (``load``) and how to
run one context/query pass (``_call``). Everything else is shared here:

* weights are loaded once and reused for every refit (``model_factory`` passes the
  loaded runner to each new instance);
* the predicted range: most of these models return a point estimate, so the spread is
  the disagreement across bootstrap resamples of the context, plus a noise floor from
  out-of-fold errors on the past runs. ``calibration.py`` then widens it if needed;
* license facts, so the UI can warn before a non-commercial model is used for real work.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ..schema import FeatureEncoder, Schema
from .base import ModelUnavailable, Prediction


def resolve_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def require(module: str, install: str):
    """Import a model's package or raise ModelUnavailable with the install hint."""
    import importlib

    try:
        return importlib.import_module(module)
    except Exception as e:
        raise ModelUnavailable(f"{module} is not installed ({e}). Install: {install}") from e


class InContextSurrogate:
    name: ClassVar[str] = "in-context"
    license: ClassVar[str] = ""
    commercial_use: ClassVar[bool | None] = None  # None = check the license yourself
    install: ClassVar[str] = ""
    # constructor arguments only used for loading weights (not passed again on refits)
    load_args: ClassVar[tuple[str, ...]] = ("ckpt", "device")

    def __init__(self, schema: Schema, n_bootstrap: int = 8, n_folds: int = 5, seed: int = 0,
                 runner: Any = None, **load_kw):
        self.schema = schema
        self.encoder = FeatureEncoder(schema)
        self.cat_idx = [i for i, v in enumerate(self.encoder.variables) if not v.is_continuous]
        self.columns = [v.name for v in self.encoder.variables]
        self.n_bootstrap = n_bootstrap
        self.n_folds = n_folds
        self.seed = seed
        self._runner = runner if runner is not None else self.load(**load_kw)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._noise_sd = 0.0

    # -- subclass hooks ---------------------------------------------------------------
    def load(self, **kw) -> Any:
        raise NotImplementedError

    def _call(self, Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        """One pass: context (Xc, yc) -> point predictions for Xq. Categoricals are integer codes."""
        raise NotImplementedError

    # -- helpers for subclasses ---------------------------------------------------------
    def frame(self, X: np.ndarray) -> pd.DataFrame:
        """Coded matrix -> DataFrame with categorical columns as pandas categories."""
        df = pd.DataFrame(X, columns=self.columns)
        for i in self.cat_idx:
            v = self.encoder.variables[i]
            df[v.name] = pd.Categorical(df[v.name].round().astype(int), categories=range(len(v.categories)))
        return df

    # -- Surrogate interface ----------------------------------------------------------------
    def _pass(self, Xc, yc, Xq) -> np.ndarray:
        out = np.asarray(self._call(Xc, yc, Xq), float).reshape(-1)
        if out.shape[0] != Xq.shape[0]:
            raise RuntimeError(f"{self.name} returned {out.shape[0]} predictions for {Xq.shape[0]} rows")
        return out

    def fit(self, data: pd.DataFrame):
        self._X, _ = self.encoder.coded(data)
        self._y = data[self.schema.target.name].to_numpy(float)
        n = len(self._y)
        if self.n_folds >= 2 and n >= 2 * self.n_folds:
            rng = np.random.default_rng(self.seed)
            resid = []
            for f in np.array_split(rng.permutation(n), self.n_folds):
                mask = np.ones(n, bool)
                mask[f] = False
                resid.append(self._y[f] - self._pass(self._X[mask], self._y[mask], self._X[f]))
            self._noise_sd = float(np.std(np.concatenate(resid)))
        else:
            self._noise_sd = float(np.std(self._y)) * 0.5
        return self

    def predict(self, rows: pd.DataFrame) -> Prediction:
        assert self._X is not None, "call fit first"
        Xq, _ = self.encoder.coded(rows)
        point = self._pass(self._X, self._y, Xq)
        rng = np.random.default_rng(self.seed + 1)
        n = len(self._y)
        boots = [point]
        for _ in range(self.n_bootstrap - 1):
            idx = rng.integers(0, n, n)
            boots.append(self._pass(self._X[idx], self._y[idx], Xq))
        spread = np.std(np.stack(boots), axis=0)
        return Prediction(point, np.sqrt(spread**2 + self._noise_sd**2))
