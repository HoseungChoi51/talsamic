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

from ..schema import Schema
from .base import ModelUnavailable
from .incontext import InContextSurrogate

HF_REPO = "stable-ai/LimiX-1_16M"
HF_FILE = "LimiX-1_16M.ckpt"


def _redirect_limix_cache() -> None:
    """LimiX v1 caches to a hard-coded /mnt/public/infe_cache, which normal users can't
    create. Point that default at $LIMIX_CACHE_DIR or ~/.cache/limix/infe_cache instead."""
    target = os.environ.get("LIMIX_CACHE_DIR", os.path.expanduser("~/.cache/limix/infe_cache"))
    try:
        from inference.v1_0 import predictor as v1
    except Exception:
        return
    def patch(cls, seen):
        if id(cls) in seen:
            return
        seen.add(id(cls))
        defaults = getattr(cls.__dict__.get("__init__"), "__defaults__", None)
        if defaults and "/mnt/public/infe_cache" in defaults:
            cls.__init__.__defaults__ = tuple(target if d == "/mnt/public/infe_cache" else d for d in defaults)
        for inner in vars(cls).values():  # CacheManager is nested inside LimiXPredictor
            if isinstance(inner, type):
                patch(inner, seen)

    seen: set[int] = set()
    for obj in vars(v1).values():
        if isinstance(obj, type) and getattr(obj, "__module__", "").startswith("inference"):
            patch(obj, seen)


def _load_predictor(device: str, ckpt: str | None, config: str | None, categorical_idx: list[int], seed: int):
    try:
        import torch
        from inference.predictor import LimiXPredictor  # installed by `pip install -e LimiX`
    except Exception as e:  # pragma: no cover - depends on the environment
        raise ModelUnavailable(f"LimiX code is not installed: {e}") from e

    for k, v in {"RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29500"}.items():
        os.environ.setdefault(k, v)
    _redirect_limix_cache()

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


class LimiXSurrogate(InContextSurrogate):
    name = "limix-16m"
    license = "Stable AI Technology License (Apache-2.0 based, attribution and naming terms)"
    commercial_use = None
    install = "git clone https://github.com/limix-ldm-ai/LimiX && pip install -e LimiX"
    load_args = ("ckpt", "config", "device", "predictor")

    def __init__(self, schema: Schema, *args, predictor=None, **kw):
        super().__init__(schema, *args, runner=kw.pop("runner", None) or predictor, **kw)

    def load(self, ckpt: str | None = None, config: str | None = None, device: str = "auto"):
        return _load_predictor(device, ckpt, config, self.cat_idx, self.seed)

    def _call(self, Xc, yc, Xq):
        # LimiX v1 (1_16M checkpoint) answers regression in standardized units: scale y
        # ourselves and map the output back (found on the RTX 5090 run, 2026-11-02).
        yc = np.asarray(yc, float)
        mu, sd = float(yc.mean()), float(yc.std()) or 1.0
        out = self._runner.predict(Xc, (yc - mu) / sd, Xq, task_type="Regression")
        return np.asarray(out, float) * sd + mu
