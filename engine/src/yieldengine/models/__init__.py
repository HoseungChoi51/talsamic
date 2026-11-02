"""Pluggable surrogate models.

``model_factory`` returns a zero-argument callable that builds a fresh, unfitted
model. Analyses refit on time splits many times, so they take a factory, not a model.
The default is LimiX-16M; if its code or weights are missing, the factory falls back
to the GP baseline and says why in ``factory.fallback_reason``.
"""

from __future__ import annotations

import warnings
from typing import Callable

from ..schema import Schema
from .base import ModelUnavailable, Prediction, Surrogate
from .gp import GPSurrogate
from .limix import LimiXSurrogate

DEFAULT_MODEL = "limix-16m"
MODELS = ("limix-16m", "gp")


class ModelFactory:
    def __init__(self, build: Callable[[], Surrogate], name: str, fallback_reason: str | None = None):
        self._build = build
        self.name = name
        self.fallback_reason = fallback_reason

    def __call__(self) -> Surrogate:
        return self._build()


def model_factory(schema: Schema, name: str = DEFAULT_MODEL, fallback: str | None = "gp", **kw) -> ModelFactory:
    if name == "gp":
        return ModelFactory(lambda: GPSurrogate(schema, **kw), "gp")
    if name == "limix-16m":
        try:
            first = LimiXSurrogate(schema, **kw)
        except ModelUnavailable as e:
            if fallback is None:
                raise
            warnings.warn(f"LimiX-16M unavailable, using {fallback}: {e}", stacklevel=2)
            fb = model_factory(schema, fallback, fallback=None)
            fb.fallback_reason = str(e)
            return fb
        shared = first._predictor  # load weights once, reuse for every refit
        kw = {k: v for k, v in kw.items() if k not in ("ckpt", "config", "device", "predictor")}
        return ModelFactory(lambda: LimiXSurrogate(schema, predictor=shared, **kw), "limix-16m")
    raise ValueError(f"unknown model {name!r}; choose from {MODELS}")


__all__ = ["DEFAULT_MODEL", "GPSurrogate", "LimiXSurrogate", "ModelFactory", "ModelUnavailable",
           "Prediction", "Surrogate", "model_factory"]
