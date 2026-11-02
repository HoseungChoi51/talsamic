"""Gaussian-process baseline: always available, honest about uncertainty on small data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from ..schema import FeatureEncoder, Schema
from .base import Prediction


class GPSurrogate:
    name = "gp"

    def __init__(self, schema: Schema, n_restarts: int = 2, seed: int = 0):
        self.schema = schema
        self.encoder = FeatureEncoder(schema)
        self.n_restarts = n_restarts
        self.seed = seed
        self._gp: GaussianProcessRegressor | None = None

    def fit(self, data: pd.DataFrame) -> "GPSurrogate":
        X = self.encoder.numeric(data)
        y = data[self.schema.target.name].to_numpy(float)
        d = X.shape[1]
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.full(d, 0.5), length_scale_bounds=(0.05, 20.0), nu=2.5
        ) + WhiteKernel(0.1, (1e-4, 1.0))
        self._gp = GaussianProcessRegressor(
            kernel, normalize_y=True, n_restarts_optimizer=self.n_restarts, random_state=self.seed
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self._gp.fit(X, y)
        return self

    def predict(self, rows: pd.DataFrame) -> Prediction:
        assert self._gp is not None, "call fit first"
        mean, sd = self._gp.predict(self.encoder.numeric(rows), return_std=True)
        return Prediction(np.asarray(mean), np.maximum(np.asarray(sd), 1e-6))
