"""Prediction benchmark: fit on a split's training rows, predict its test rows.

Reports accuracy (RMSE, MAE, R²) and whether the 80% ranges are honest (coverage should
be near 80%; width says how useful they are). Ranges here are the model's own, before
the calibration step, so the coverage column shows how much calibration each model needs.
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from ..models import model_factory
from .datasets import Benchmark

GP_MAX_CONTEXT = 500  # exact GP with ARD gets slow fast (n³ per optimizer step); subsample above this


def evaluate(bench: Benchmark, split: str, model: str, max_context: int | None = None, seed: int = 0,
             **model_kw) -> dict:
    train, test = bench.split(split)
    cap = max_context or (GP_MAX_CONTEXT if model == "gp" else None)
    note = ""
    if cap and len(train) > cap:
        train = train.sample(cap, random_state=seed).reset_index(drop=True)
        note = f"context subsampled to {cap}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factory = model_factory(bench.schema, model, fallback=None, **model_kw)
        t = time.time()
        p = factory().fit(train).predict(test)
        secs = time.time() - t
    y = test[bench.schema.target.name].to_numpy(float)
    err = y - p.mean
    lo, hi = p.interval(0.8)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "benchmark": bench.name, "split": split, "model": model, "n_train": len(train), "n_test": len(test),
        "rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))),
        "r2": 1 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else float("nan"),
        "coverage80": float(np.mean((y >= lo) & (y <= hi))), "width80": float(np.mean(hi - lo)),
        "seconds": secs, "note": note,
    }


def results_table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    cols = ["benchmark", "split", "model", "n_train", "rmse", "r2", "coverage80", "width80", "seconds"]
    return df[cols].round({"rmse": 2, "r2": 3, "coverage80": 2, "width80": 1, "seconds": 1}).to_markdown(index=False)
