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

from ..schema import Schema
from .base import ModelUnavailable
from .incontext import InContextSurrogate

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
        return self._runner.predict(Xc, yc, Xq, task_type="Regression")
