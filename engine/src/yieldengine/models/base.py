"""The model interface every surrogate implements.

Analyses and suggestions only ever call ``fit`` and ``predict``, so swapping
LimiX for a GP (or TabPFN later, with a commercial license) changes nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..schema import Schema


@dataclass
class Prediction:
    """Predicted yield as a normal distribution per row."""

    mean: np.ndarray
    sd: np.ndarray

    def quantile(self, q: float) -> np.ndarray:
        return self.mean + norm.ppf(q) * self.sd

    def interval(self, coverage: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
        a = (1 - coverage) / 2
        return self.quantile(a), self.quantile(1 - a)

    def __len__(self) -> int:
        return len(self.mean)


@runtime_checkable
class Surrogate(Protocol):
    name: str
    schema: Schema

    def fit(self, data: pd.DataFrame) -> "Surrogate": ...

    def predict(self, rows: pd.DataFrame) -> Prediction: ...


class ModelUnavailable(RuntimeError):
    """Raised when a model's code or weights cannot be loaded in this environment."""
