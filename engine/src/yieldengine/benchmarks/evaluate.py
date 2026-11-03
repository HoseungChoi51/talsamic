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
from ..novelty import guarded, new_conditions
from .datasets import Benchmark

GP_MAX_CONTEXT = 500  # exact GP with ARD gets slow fast (n³ per optimizer step); subsample above this


def evaluate(bench: Benchmark, split: str, model: str, max_context: int | None = None, seed: int = 0,
             guard: bool = False, **model_kw) -> dict:
    """``guard``: wrap the model in the new-chemical guard (novelty.py). Either way the
    result splits coverage into test rows with a value the training rows never used and the rest."""
    train, test = bench.split(split)
    cap = max_context or (GP_MAX_CONTEXT if model == "gp" else None)
    note = ""
    if cap and len(train) > cap:
        train = train.sample(cap, random_state=seed).reset_index(drop=True)
        note = f"context subsampled to {cap}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        factory = model_factory(bench.schema, model, fallback=None, **model_kw)
        if guard:
            factory = guarded(factory, bench.schema)
        t = time.time()
        p = factory().fit(train).predict(test)
        secs = time.time() - t
    y = test[bench.schema.target.name].to_numpy(float)
    err = y - p.mean
    lo, hi = p.interval(0.8)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    inside = (y >= lo) & (y <= hi)
    new = np.array([bool(n) for n in new_conditions(bench.schema, train, test)])
    return {
        "benchmark": bench.name, "split": split, "model": model + bench.name_suffix + ("+guard" if guard else ""),
        "n_train": len(train), "n_test": len(test),
        "rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err))),
        "r2": 1 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else float("nan"),
        "coverage80": float(np.mean(inside)), "width80": float(np.mean(hi - lo)),
        "share_new": float(np.mean(new)),
        "coverage80_new": float(np.mean(inside[new])) if new.any() else float("nan"),
        "coverage80_seen": float(np.mean(inside[~new])) if (~new).any() else float("nan"),
        "seconds": secs, "note": note,
    }


def results_table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    cols = ["benchmark", "split", "model", "n_train", "rmse", "r2", "coverage80", "coverage80_new", "width80", "seconds"]
    return df[cols].round({"rmse": 2, "r2": 3, "coverage80": 2, "coverage80_new": 2, "width80": 1,
                           "seconds": 1}).to_markdown(index=False)
