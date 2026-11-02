"""Optimization benchmark: how fast does the whole suggestion pipeline find high yields?

A campaign replays what the tool does at a plant, with a benchmark dataset as the plant:
start from a few known runs, then each round fit the model, ask ``suggest_batch`` for a
batch from the untested recipes, "run" them by looking up their measured yields, and
repeat. The dataset holds every combination, so every suggestion has a real answer.

    python -m yieldengine.benchmarks.campaign --models random gp exaone-tabular \
        --benchmarks direct_arylation --rounds 8 --batch 8 --seeds 0 1 2

Reported per round: best yield found so far, its gap to the best in the dataset, and
how many of the top 1% recipes have been found.
"""

from __future__ import annotations

import argparse
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ..models import MODELS, model_factory
from ..novelty import guarded
from ..suggest import BatchSettings, suggest_batch
from .datasets import LOADERS, Benchmark, load


def _pool(bench: Benchmark) -> pd.DataFrame:
    cols = [v.name for v in bench.schema.controls]
    return bench.data.groupby(cols, as_index=False, observed=True)["yield"].mean()


def _initial(pool: pd.DataFrame, n: int, start: str, rng) -> np.ndarray:
    if start == "low":  # a plant that has only run mediocre recipes so far
        below = np.where(pool["yield"].to_numpy() <= pool["yield"].median())[0]
        return rng.choice(below, n, replace=False)
    return rng.choice(len(pool), n, replace=False)


def run_campaign(bench: Benchmark, model: str, rounds: int = 8, batch: int = 8, learn: int = 2,
                 n_init: int = 10, start: str = "random", risk: str = "균형", radius: float | None = None,
                 seed: int = 0, guard: bool = False, **model_kw) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = _pool(bench)
    cols = [v.name for v in bench.schema.controls]
    y_all = pool["yield"].to_numpy()
    top = y_all >= np.quantile(y_all, 0.99)
    tested = np.zeros(len(pool), bool)
    tested[_initial(pool, n_init, start, rng)] = True
    key = pool[cols].astype(str).agg("|".join, axis=1)
    index_of = dict(zip(key, range(len(pool))))

    factory = None
    if model != "random":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            factory = model_factory(bench.schema, model, fallback=None, **model_kw)
            if guard:
                factory = guarded(factory, bench.schema)
    settings = BatchSettings(risk=risk, n_runs=batch, split=(batch - learn, learn, 0),
                             max_changes=len(cols), radius=radius, seed=seed)

    def record(r, secs):
        y = y_all[tested]
        label = model + bench.name_suffix + ("+guard" if guard and model != "random" else "")
        return {"benchmark": bench.name, "model": label, "seed": seed, "start": start, "risk": risk,
                "round": r, "n_runs": int(tested.sum()), "best": float(y.max()),
                "gap_to_best": float(y_all.max() - y.max()), "top1pct_found": int((tested & top).sum()),
                "top1pct_total": int(top.sum()), "seconds": secs}

    rows = [record(0, 0.0)]
    for r in range(1, rounds + 1):
        t = time.time()
        untested = np.where(~tested)[0]
        if model == "random":
            pick = rng.choice(untested, min(batch, len(untested)), replace=False)
        else:
            observed = pool[tested].reset_index(drop=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = factory().fit(observed)
                b = suggest_batch(m, observed, bench.schema, settings, candidates=pool.iloc[untested])
            pick = [index_of["|".join(str(s.recipe[c]) for c in cols)] for s in b.suggestions]
        tested[pick] = True
        rows.append(record(r, time.time() - t))
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["random", "gp"], choices=("random",) + MODELS)
    ap.add_argument("--benchmarks", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--learn", type=int, default=2)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--start", default="random", choices=["random", "low"])
    ap.add_argument("--risk", default="균형", choices=["안전하게", "균형", "도전적으로"])
    ap.add_argument("--radius", type=float)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-bootstrap", type=int, default=4)
    ap.add_argument("--out", default="results/campaign.csv")
    ap.add_argument("--descriptors", help="folder of LLM descriptor tables (<benchmark>.json)")
    ap.add_argument("--guard", action="store_true", help="wrap models in the new-chemical guard (novelty.py)")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    finals = []
    for bname in a.benchmarks:
        bench = load(bname, a.descriptors)
        for m in a.models:
            kw = {} if m == "random" else ({"n_restarts": 0} if m == "gp" else
                                           {"device": a.device, "n_bootstrap": a.n_bootstrap})
            for seed in a.seeds:
                try:
                    df = run_campaign(bench, m, a.rounds, a.batch, a.learn, a.n_init, a.start, a.risk,
                                      a.radius, seed, guard=a.guard, **kw)
                except Exception as e:
                    traceback.print_exc()
                    print(f"{bname} {m} seed {seed}: ERROR {type(e).__name__}: {e}", flush=True)
                    continue
                df.to_csv(out, mode="a", header=not out.exists(), index=False)
                last = df.iloc[-1]
                finals.append(last)
                print(f"{bname:17s} {m:15s} seed {seed}: best {last.best:5.1f} (gap {last.gap_to_best:4.1f}), "
                      f"top-1% found {last.top1pct_found}/{last.top1pct_total}, {df.seconds.sum():.0f}s", flush=True)
    if finals:
        f = pd.DataFrame(finals)
        print("\n" + f.groupby(["benchmark", "model"])[["best", "gap_to_best", "top1pct_found"]]
              .mean().round(1).to_markdown())


if __name__ == "__main__":
    main()
