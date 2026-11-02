"""Pluggable surrogate models.

``model_factory`` returns a zero-argument callable that builds a fresh, unfitted
model. Analyses refit on time splits many times, so they take a factory, not a model.
Weights are loaded once by the factory and shared by every model it builds.
The default is TabFM 1.1, the strongest optimizer in the benchmarks. Its weights are
non-commercial, so ``license_note`` gives the UI a sentence to show until the plant
has a license (or picks ``limix-16m`` or ``gp``). If a model's code or weights are
missing, the factory falls back to the GP baseline and says why in
``factory.fallback_reason``.
"""

from __future__ import annotations

import warnings
from typing import Callable

from ..schema import Schema
from .base import ModelUnavailable, Prediction, Surrogate
from .foundation import EXAONETabularSurrogate, MitraSurrogate, TabFMSurrogate, TabLDMSurrogate
from .gp import GPSurrogate
from .incontext import InContextSurrogate
from .limix import LimiXSurrogate

DEFAULT_MODEL = "tabfm"
REGISTRY: dict[str, type] = {
    "limix-16m": LimiXSurrogate,
    "exaone-tabular": EXAONETabularSurrogate,
    "tabfm": TabFMSurrogate,
    "mitra": MitraSurrogate,
    "tabldm": TabLDMSurrogate,
    "gp": GPSurrogate,
}
MODELS = tuple(REGISTRY)


class ModelFactory:
    def __init__(self, build: Callable[[], Surrogate], name: str, fallback_reason: str | None = None):
        self._build = build
        self.name = name
        self.fallback_reason = fallback_reason

    def __call__(self) -> Surrogate:
        return self._build()


def model_factory(schema: Schema, name: str = DEFAULT_MODEL, fallback: str | None = "gp", **kw) -> ModelFactory:
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {MODELS}")
    cls = REGISTRY[name]
    if not issubclass(cls, InContextSurrogate):
        return ModelFactory(lambda: cls(schema, **kw), name)
    try:
        first = cls(schema, **kw)
    except ModelUnavailable as e:
        if fallback is None:
            raise
        warnings.warn(f"{name} unavailable, using {fallback}: {e}", stacklevel=2)
        fb = model_factory(schema, fallback, fallback=None)
        fb.fallback_reason = f"{name}: {e}"
        return fb
    shared = first._runner  # load weights once, reuse for every refit
    rest = {k: v for k, v in kw.items() if k not in cls.load_args and k != "runner"}
    return ModelFactory(lambda: cls(schema, runner=shared, **rest), name)


def model_info() -> list[dict]:
    """License and install facts for the UI's model picker."""
    out = []
    for name, cls in REGISTRY.items():
        out.append({"name": name, "license": getattr(cls, "license", "BSD-3-Clause (scikit-learn)"),
                    "commercial_use": getattr(cls, "commercial_use", True),
                    "install": getattr(cls, "install", "included")})
    return out


def license_note(name: str) -> str | None:
    """A Korean sentence for the UI when a model's license may not allow plant use, else None."""
    cls = REGISTRY[name]
    ok = getattr(cls, "commercial_use", True)
    if ok:
        return None
    label = getattr(cls, "label", name)
    if ok is None:
        return f"{label}의 라이선스 조건은 아직 확인하지 않았다. 실제 생산에 쓰기 전에 확인이 필요하다."
    return f"{label}의 가중치는 비상업 라이선스라서 실제 생산에 쓰려면 별도 라이선스가 필요하다."


__all__ = ["DEFAULT_MODEL", "EXAONETabularSurrogate", "GPSurrogate", "InContextSurrogate", "LimiXSurrogate",
           "MODELS", "MitraSurrogate", "ModelFactory", "ModelUnavailable", "Prediction", "REGISTRY", "Surrogate",
           "TabFMSurrogate", "TabLDMSurrogate", "license_note", "model_factory", "model_info"]
