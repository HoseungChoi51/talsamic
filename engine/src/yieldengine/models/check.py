"""Which models can run on this machine?

    python -m yieldengine.models.check            # import check only
    python -m yieldengine.models.check --load     # also load weights (downloads them)
    python -m yieldengine.models.check --load --smoke   # and fit/predict on the fake plant
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from ..simulator import FakePlant, default_schema
from . import REGISTRY, ModelUnavailable, model_info
from .incontext import InContextSurrogate


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args(argv)
    schema = default_schema()
    hist = FakePlant(seed=0).history()
    train, test = hist.iloc[:100], hist.iloc[100:]
    info = {m["name"]: m for m in model_info()}
    for name, cls in REGISTRY.items():
        if a.only and name not in a.only:
            continue
        row = f"{name:15s} commercial={str(info[name]['commercial_use']):5s} "
        if not issubclass(cls, InContextSurrogate):
            row += "ok (built in)"
        elif not a.load:
            try:
                mod = {"limix-16m": "inference.predictor", "exaone-tabular": "exaonetabular", "tabfm": "tabfm",
                       "mitra": "autogluon.tabular", "tabldm": "tabldm"}[name]
                __import__(mod)
                row += "package installed (weights not checked; use --load)"
            except Exception:
                row += f"NOT installed -> {cls.install}"
        else:
            t = time.time()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = cls(schema, device=a.device, n_bootstrap=2, n_folds=0)
                row += f"loaded in {time.time() - t:.0f}s"
                if a.smoke:
                    t = time.time()
                    p = m.fit(train).predict(test)
                    rmse = float(np.sqrt(np.mean((p.mean - test['yield'].to_numpy()) ** 2)))
                    row += f", smoke RMSE {rmse:.2f} on 40 runs in {time.time() - t:.0f}s"
            except ModelUnavailable as e:
                row += f"unavailable: {e}"
            except Exception as e:  # report, don't crash the whole check
                row += f"ERROR {type(e).__name__}: {e}"
        print(row, flush=True)


if __name__ == "__main__":
    main()
