"""Run the prediction benchmark.

    python -m yieldengine.benchmarks.run --models gp exaone-tabular tabfm --benchmarks direct_arylation \
        --splits n100 ligand_out1 --out results/prediction.csv

Without --splits every split of each benchmark runs. Results are appended to --out,
so an interrupted run can be resumed with --skip-done.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import pandas as pd

from ..models import MODELS
from .datasets import LOADERS, load
from .evaluate import evaluate, results_table


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gp"], choices=MODELS)
    ap.add_argument("--benchmarks", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--splits", nargs="*")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n-bootstrap", type=int, default=8)
    ap.add_argument("--max-context", type=int)
    ap.add_argument("--out", default="results/prediction.csv")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.skip_done and out.exists():
        prev = pd.read_csv(out)
        done = set(zip(prev.benchmark, prev.split, prev.model))
    rows = []
    for bname in a.benchmarks:
        bench = load(bname)
        for split in a.splits or list(bench.splits):
            if split not in bench.splits:
                continue
            for m in a.models:
                if (bname, split, m) in done:
                    continue
                kw = {"n_restarts": 0} if m == "gp" else {"device": a.device, "n_bootstrap": a.n_bootstrap}
                try:
                    r = evaluate(bench, split, m, max_context=a.max_context, **kw)
                except Exception as e:
                    traceback.print_exc()
                    r = {"benchmark": bname, "split": split, "model": m, "note": f"ERROR {type(e).__name__}: {e}"}
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                rows.append(r)
                pd.DataFrame([r]).to_csv(out, mode="a", header=not out.exists(), index=False)
    ok = [r for r in rows if "rmse" in r]
    if ok:
        print("\n" + results_table(ok))


if __name__ == "__main__":
    main()
