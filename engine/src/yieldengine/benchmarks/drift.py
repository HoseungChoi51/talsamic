"""Drift benchmark on the fake plant: does the drift correction put predictions at today's level?

For each seed, the fake plant's history ends with a drop (``--drift``, default 2%p; 0 checks
that nothing switches on without drift). The engine is fitted with and without the
correction, and two sets of runs are made "today":

* old: every recipe from before the drift, rerun at today's lot and day. This is the case
  the correction is for: the model knows these recipes only at the old level;
* batch: the 12-run batch the engine suggests, as in the demo's 이번 주 check. Suggested runs
  are picked where the model is optimistic, so they also come in low without any drift.

    python -m yieldengine.benchmarks.drift --model tabfm --seeds 0 1 2 3 --drift 2 0
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

from ..engine import Engine
from ..simulator import FakePlant, default_schema
from ..suggest import BatchSettings


def _check(model, runs: pd.DataFrame, y: np.ndarray) -> dict:
    p = model.predict(runs)
    lo, hi = p.interval(0.8)
    return {"bias": float(np.mean(y - p.mean)), "coverage80": float(np.mean((y >= lo) & (y <= hi))),
            "width80": float(np.mean(hi - lo))}


def run_drift(model: str = "gp", seeds=(0, 1, 2, 3), drifts=(2.0, 0.0), **model_kw) -> pd.DataFrame:
    schema = default_schema()
    rows = []
    for drift in drifts:
        for seed in seeds:
            for corr in (False, True):
                plant = FakePlant(seed=seed, drift_size=drift)
                h = plant.history()
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eng = Engine(h, schema, model=model, context=plant.context_now(h), drift_correction=corr, **model_kw)
                    batch = eng.suggest(BatchSettings(n_runs=12, seed=seed)).to_frame()
                old = h[h["days"] < plant.drift_start_day - 21].drop(columns=["yield", "peak_temp", "date"])
                old = old.assign(**eng.context)
                ran_old, ran_batch = plant.run(old), plant.run(batch)
                row = {"model": eng.model_name, "drift": drift, "seed": seed, "correction": corr,
                       "shift": eng.model.shift, "measured": eng.drift.delta if eng.drift else np.nan}
                for name, runs in (("old", ran_old), ("batch", ran_batch)):
                    row.update({f"{name}_{k}": v for k, v in _check(eng.model, runs, runs["yield"].to_numpy()).items()})
                row["seconds"] = time.time() - t0
                rows.append(row)
                print({k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}, flush=True)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gp")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--drift", type=float, nargs="+", default=[2.0, 0.0])
    ap.add_argument("--device", default=None, help="for the foundation models, e.g. cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    kw = {"device": a.device} if a.device else {}
    df = run_drift(a.model, a.seeds, a.drift, **kw)
    cols = ["shift", "old_bias", "old_coverage80", "old_width80", "batch_bias", "batch_coverage80", "seconds"]
    print(df.groupby(["model", "drift", "correction"])[cols].mean().round(2).to_markdown())
    if a.out:
        df.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
