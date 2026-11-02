import numpy as np
import pandas as pd

from yieldengine.benchmarks.campaign import run_campaign
from yieldengine.benchmarks.datasets import Benchmark, _group_out, _schema, _standard_splits
from yieldengine.suggest import BatchSettings, filter_pool


def _toy(seed=0) -> Benchmark:
    rng = np.random.default_rng(seed)
    grid = pd.MultiIndex.from_product([["L1", "L2", "L3", "L4"], ["B1", "B2"], [80.0, 100.0, 120.0]],
                                      names=["ligand", "base", "temp"]).to_frame(index=False)
    best = {"L1": 10, "L2": 30, "L3": 60, "L4": 20}
    grid["yield"] = (grid["ligand"].map(best) + 10 * (grid["base"] == "B2") - 0.01 * (grid["temp"] - 110) ** 2
                     + rng.normal(0, 1, len(grid))).clip(0, 100)
    schema = _schema(grid, {"ligand": "리간드", "base": "염기"}, {"temp": ("온도", "℃")})
    return Benchmark("toy", grid, schema, {**_standard_splits(grid), **_group_out(grid, "ligand", 2)})


def test_group_out_holds_out_whole_categories():
    b = _toy()
    for name in ("ligand_out1", "ligand_out2"):
        tr, te = b.split(name)
        assert set(tr.ligand).isdisjoint(te.ligand) and len(tr) + len(te) == len(b.data)


def test_filter_pool_respects_constraints():
    b = _toy()
    s = BatchSettings(fixed={"base": "B2"}, limits={"temp": (90, 130), "ligand": ["L1", "L3"]}, max_changes=1)
    pool = filter_pool(b.schema, s, b.data)
    assert len(pool) > 0
    assert (pool.base == "B2").all() and pool.temp.min() >= 90 and set(pool.ligand) <= {"L1", "L3"}


def test_campaign_beats_random_on_toy():
    b = _toy()
    gp = run_campaign(b, "gp", rounds=3, batch=4, learn=1, n_init=4, start="low", seed=0, n_restarts=0)
    assert list(gp["round"]) == [0, 1, 2, 3] and gp["n_runs"].iloc[-1] == 16
    assert gp["best"].is_monotonic_increasing
    assert gp["gap_to_best"].iloc[-1] <= 5  # finds L3/B2 near the optimum within 12 runs
