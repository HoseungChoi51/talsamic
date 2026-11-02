"""LimiX-16M adapter (default model).

LimiX is an in-context model: ``fit`` only stores the past runs, and ``predict``
scores every query row in one forward pass with those runs as context.

Its regression output is a point estimate, so the spread comes from two parts:
* model spread: the same queries scored against several bootstrap resamples of the context;
* noise floor: out-of-fold error on the past runs (one extra pass per fold).
The calibration step in ``calibration.py`` then widens the result using
time-ordered errors, as for any model.

Setup (not done automatically):
    git clone https://github.com/limix-ldm-ai/LimiX && pip install -e LimiX
    export YIELDENGINE_LIMIX_CKPT=/path/to/LimiX-16M.ckpt      # or let huggingface_hub fetch it
    export YIELDENGINE_LIMIX_CONFIG=/path/to/LimiX/config/reg_default_noretrieval.json
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import FeatureEncoder, Schema
from .base import ModelUnavailable, Prediction

HF_REPO = "stable-ai/LimiX-1_16M"
HF_FILE = "LimiX-16M.ckpt"


def _load_predictor(device: str, ckpt: str | None, config: str | None, categorical_idx: list[int], seed: int):
    try:
        import torch
        from inference.predictor import LimiXPredictor  # installed by `pip install -e LimiX`
    except Exception as e:  # pragma: no cover - depends on the environment
        raise ModelUnavailable(f"LimiX code is not installed: {e}") from e

    for k, v in {"RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}.items():
        os.environ.setdefault(k, v)

    ckpt = ckpt or os.environ.get("YIELDENGINE_LIMIX_CKPT")
    if not ckpt:
        try:
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)
        except Exception as e:  # pragma: no cover
            raise ModelUnavailable(f"LimiX-16M weights could not be downloaded: {e}") from e

    config = config or os.environ.get("YIELDENGINE_LIMIX_CONFIG")
    if not config:
        import inference

        guess = Path(inference.__file__).resolve().parent.parent / "config" / "reg_default_noretrieval.json"
        if not guess.exists():
            raise ModelUnavailable("set YIELDENGINE_LIMIX_CONFIG to LimiX's reg_default_noretrieval.json")
        config = str(guess)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return LimiXPredictor(
        device=torch.device(device),
        model_path=ckpt,
        inference_config=config,
        categorical_features_indices=categorical_idx or None,
        seed=seed,
    )


class LimiXSurrogate:
    name = "limix-16m"

    def __init__(
        self,
        schema: Schema,
        ckpt: str | None = None,
        config: str | None = None,
        device: str = "auto",
        n_bootstrap: int = 8,
        n_folds: int = 5,
        seed: int = 0,
        predictor=None,  # inject a stand-in for tests
    ):
        self.schema = schema
        self.encoder = FeatureEncoder(schema)
        self.n_bootstrap = n_bootstrap
        self.n_folds = n_folds
        self.seed = seed
        cat_idx = [i for i, v in enumerate(self.encoder.variables) if not v.is_continuous]
        self._predictor = predictor or _load_predictor(device, ckpt, config, cat_idx, seed)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._noise_sd = 0.0

    def _call(self, Xc: np.ndarray, yc: np.ndarray, Xq: np.ndarray) -> np.ndarray:
        return np.asarray(self._predictor.predict(Xc, yc, Xq, task_type="Regression"), float).reshape(-1)

    def fit(self, data: pd.DataFrame) -> "LimiXSurrogate":
        self._X, _ = self.encoder.coded(data)
        self._y = data[self.schema.target.name].to_numpy(float)
        n = len(self._y)
        rng = np.random.default_rng(self.seed)
        folds = np.array_split(rng.permutation(n), min(self.n_folds, n))
        resid = []
        for f in folds:
            mask = np.ones(n, bool)
            mask[f] = False
            resid.append(self._y[f] - self._call(self._X[mask], self._y[mask], self._X[f]))
        self._noise_sd = float(np.std(np.concatenate(resid)))
        return self

    def predict(self, rows: pd.DataFrame) -> Prediction:
        assert self._X is not None, "call fit first"
        Xq, _ = self.encoder.coded(rows)
        point = self._call(self._X, self._y, Xq)
        rng = np.random.default_rng(self.seed + 1)
        n = len(self._y)
        boots = [point]
        for _ in range(self.n_bootstrap - 1):
            idx = rng.integers(0, n, n)
            boots.append(self._call(self._X[idx], self._y[idx], Xq))
        spread = np.std(np.stack(boots), axis=0)
        return Prediction(point, np.sqrt(spread**2 + self._noise_sd**2))
