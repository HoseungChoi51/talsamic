"""Traps case study: what a tabular foundation model gives you on ~700 synthesis runs,
and where it bites. Every experiment runs on the fake synthesis in ``synthesis.py``,
where the true mechanism is known, so each claim is graded against the truth.

    python -m yieldengine.studies.traps --models gp gbdt --out results/traps
    python -m yieldengine.studies.traps --models tabfm exaone-tabular limix-16m --out results/traps
    python -m yieldengine.studies.traps --models mitra --experiments accuracy extrapolation encoding

One JSON per model is written to --out. ``report`` experiments that need no model
(data summaries, sparse regression for the mechanism) run with --models none.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import synthesis as S
from .models import LEVELS, Dist, Timer, build

SEED = 0


# ------------------------------------------------------------------------------ data
def history(seed=SEED, **kw):
    h = S.make_history(seed=seed, **kw)
    d = h.data.copy()
    d["promoter_used"] = (h.truth["true_promoter"] > 0).to_numpy()  # from the recipe sheet
    return d, h.truth


def fresh(rec: pd.DataFrame, rng, lab="A", observe=True):
    """Turn true recipes into rows as they would be logged, plus their truth."""
    d = rec.copy().reset_index(drop=True)
    d["promoter_used"] = d["promoter_wt"] > 0
    truth = pd.DataFrame({"true_promoter": d["promoter_wt"].to_numpy(),
                          "true_yield": S.true_yield(d, S.AVG_PURITY)})
    if observe:
        purity = rng.uniform(0.72, 1.0, len(d))
        y, failed = S.observe(d, purity, rng)
        d["yield"] = y
        truth["failed"] = failed
        truth["purity"] = purity
    d["promoter_wt"] = np.where(d["promoter_used"], d["promoter_wt"], np.nan)
    d["lab"] = lab
    return d, truth


def near_history(d: pd.DataFrame, n: int, rng) -> pd.DataFrame:
    """Recipes like the history's (jittered copies of its rows): the interior."""
    base = d.sample(n, replace=True, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    tp = np.where(base["promoter_used"], base["promoter_wt"].fillna(2.5), 0.0)
    return pd.DataFrame({
        "catalyst": base["catalyst"], "T_C": np.clip(base["T_C"] + rng.normal(0, 6, n), 110, S.HIST_T_MAX),
        "P_bar": np.clip(base["P_bar"] * np.exp(rng.normal(0, 0.15, n)), 1, 50),
        "t_h": np.clip(base["t_h"] * np.exp(rng.normal(0, 0.2, n)), 0.25, 40),
        "cat_mol": np.clip(base["cat_mol"] * np.exp(rng.normal(0, 0.15, n)), 0.5, 10),
        "promoter_wt": tp, "solvent": base["solvent"], "stir_rpm": base["stir_rpm"]})


ENGINEERED = dict(labels=True, desc=True, transformed=True, promoter="flag")


def X(df, **kw):
    return S.features(df, **{**ENGINEERED, **kw})


# --------------------------------------------------------------------------- metrics
def metrics(y, d: Dist | None = None, point=None) -> dict:
    y = np.asarray(y, float)
    m = d.mean if d is not None else np.asarray(point, float)
    out = {"n": int(len(y)), "rmse": float(np.sqrt(np.mean((m - y) ** 2))), "mae": float(np.mean(np.abs(m - y))),
           "r2": float(1 - np.sum((m - y) ** 2) / np.sum((y - y.mean()) ** 2)) if len(y) > 1 else None,
           "bias": float(np.mean(m - y))}
    if d is not None:
        for c in (0.5, 0.8, 0.95):
            lo, hi = d.interval(c)
            out[f"cov{int(c * 100)}"] = float(np.mean((y >= lo) & (y <= hi)))
            out[f"width{int(c * 100)}"] = float(np.mean(hi - lo))
        # pinball loss over the 99 levels approximates the CRPS / 2
        diff = y[:, None] - d.q
        out["crps"] = float(2 * np.mean(np.maximum(LEVELS * diff, (LEVELS - 1) * diff)))
        out["pit_curve"] = [float(np.mean(y <= d.q[:, k])) for k in range(4, 99, 5)]  # levels .05..95
        out["below0"] = float(np.mean(d.q[:, 49] < 0))
        out["above1"] = float(np.mean(d.q[:, 49] > 1))
        out["q10_below0"] = float(np.mean(d.q[:, 9] < 0))
        out["q90_above1"] = float(np.mean(d.q[:, 89] > 1))
        out["dist_source"] = d.source
    return out


def group_folds(groups, k, seed=SEED):
    g = pd.Series(groups).astype(str)
    uniq = g.unique()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of = {uniq[j]: i % k for i, j in enumerate(perm)}
    f = g.map(fold_of).to_numpy()
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]


def random_folds(n, k, seed=SEED):
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(n), k)
    return [(np.setdiff1d(np.arange(n), p), np.sort(p)) for p in parts]


def cv(model, F, cats, y, folds, kind="dist", timer=None, key="cv"):
    n = len(y)
    mean = np.full(n, np.nan)
    q = np.full((n, len(LEVELS)), np.nan)
    src = None
    for tr, te in folds:
        t0 = time.perf_counter()
        if kind == "dist":
            d = model.dist(F.iloc[tr], cats, y[tr], F.iloc[te])
            mean[te], q[te], src = d.mean, d.q, d.source
        else:
            mean[te] = model.point(F.iloc[tr], cats, y[tr], F.iloc[te])
        if timer is not None:
            timer.t.setdefault(key, []).append(time.perf_counter() - t0)
    return Dist(mean, q, src or "point") if kind == "dist" else mean


def r(x, k=4):
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [r(v, k) for v in np.asarray(x, dtype=float).tolist()]
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), k)


# ======================================================================== experiments
def exp_accuracy(model, timer):
    """Expectation 1 + trap 5: grouped vs random vs time-ordered CV, with and without a
    leaked measured column. Calibration curves for the grouped split."""
    d, _ = history()
    y = d["yield"].to_numpy(float)
    F, cats = X(d)
    out = {}
    splits = {"grouped": group_folds(d["batch"], 5), "random": random_folds(len(d), 5)}
    cut = int(0.8 * len(d))
    splits["time"] = [(np.arange(cut), np.arange(cut, len(d)))]
    preds = {}
    for name, folds in splits.items():
        dd = cv(model, F, cats, y, folds, "dist", timer, f"cv_{name}")
        idx = np.concatenate([te for _, te in folds])
        out[name] = metrics(y[idx], dd.take(idx))
        preds[name] = dd
    g = preds["grouped"]
    lo, hi = g.interval(0.8)
    out["grouped_rows"] = {"y": r(y, 3), "mean": r(g.mean, 3), "lo80": r(lo, 3), "hi80": r(hi, 3),
                           "batch": d["batch"].tolist()}
    if hasattr(model, "native") and model.name in ("exaone-tabular", "limix-16m") and model.dist_source == "native":
        folds = splits["grouped"]
        mean = np.full(len(y), np.nan)
        q = np.full((len(y), len(LEVELS)), np.nan)
        for tr, te in folds:
            dd = model.dist(F.iloc[tr], cats, y[tr], F.iloc[te], native=False)
            mean[te], q[te] = dd.mean, dd.q
        out["grouped_bootstrap"] = metrics(y, Dist(mean, q, "bootstrap"))
    FL, cl = X(d, leak=True)
    for name in ("random", "grouped"):
        p = cv(model, FL, cl, y, splits[name], "point", timer, f"leak_{name}")
        out[f"leak_{name}"] = metrics(y, point=p)
    return out


def exp_extrapolation(model, timer):
    """Trap 1: calibration inside vs outside the history; a temperature slice to 300 C."""
    d, _ = history()
    y = d["yield"].to_numpy(float)
    rng = np.random.default_rng(11)
    inner, ti = fresh(near_history(d, 400, rng), rng)
    edge_rec = near_history(d, 400, rng)
    edge_rec["T_C"] = rng.uniform(242, 300, len(edge_rec))
    edge, te = fresh(edge_rec, rng)
    unseen_rec = near_history(d, 300, rng)
    unseen_rec["catalyst"] = rng.choice(list(S.UNSEEN), len(unseen_rec))
    unseen, tu = fresh(unseen_rec, rng)
    # temperature slice at a good recipe from the history
    base = {"catalyst": "PD-C2", "P_bar": 20.0, "t_h": 8.0, "cat_mol": 4.0, "promoter_wt": 2.5,
            "solvent": "ethanol", "stir_rpm": 500.0}
    Ts = np.arange(100, 301, 5.0)
    sl_rec = pd.DataFrame([{**base, "T_C": T} for T in Ts])
    sl, ts = fresh(sl_rec, rng, observe=False)
    allq = pd.concat([inner, edge, unseen, sl], ignore_index=True)
    F, cats = X(pd.concat([d, allq], ignore_index=True))
    Ftr, Fq = F.iloc[: len(d)], F.iloc[len(d):]
    with timer("extrapolation"):
        dd = model.dist(Ftr, cats, y, Fq)
    n1, n2, n3 = len(inner), len(edge), len(unseen)
    parts = {"interior": (slice(0, n1), inner, ti), "edge_T": (slice(n1, n1 + n2), edge, te),
             "unseen_catalyst": (slice(n1 + n2, n1 + n2 + n3), unseen, tu)}
    out = {}
    for k, (s, rows, tr) in parts.items():
        sub = dd.take(np.arange(len(allq))[s])
        out[k] = metrics(rows["yield"].to_numpy(float), sub)
        out[k]["mean_sd"] = float(np.mean(sub.sd))
        out[k]["truth_rmse"] = float(np.sqrt(np.mean((sub.mean - tr["true_yield"].to_numpy()) ** 2)))
    # sd against temperature for the edge set (does uncertainty widen past 240 C?)
    se = dd.take(np.arange(n1, n1 + n2))
    out["edge_points"] = {"T": r(edge["T_C"], 1), "sd": r(se.sd, 4), "err": r(se.mean - edge["yield"].to_numpy(), 4)}
    si = dd.take(np.arange(n1 + n2 + n3, len(allq)))
    lo80, hi80 = si.interval(0.8)
    lo95, hi95 = si.interval(0.95)
    out["slice"] = {"T": r(Ts, 1), "mean": r(si.mean), "lo80": r(lo80), "hi80": r(hi80), "lo95": r(lo95),
                    "hi95": r(hi95), "truth": r(ts["true_yield"]),
                    "expected": r(ts["true_yield"] * (1 - S.fail_prob(sl_rec)) * 1.0), "recipe": base}
    return out


def exp_encoding(model, timer):
    """Trap 2: labels vs descriptors, raw vs transformed coordinates, three splits."""
    d, _ = history()
    y = d["yield"].to_numpy(float)
    variants = {"labels_raw": dict(desc=False, transformed=False), "labels_transformed": dict(desc=False),
                "descriptors_raw": dict(transformed=False), "descriptors_transformed": {}}
    splits = {"grouped": group_folds(d["batch"], 5), "leave_catalyst_out": group_folds(d["catalyst"], 7, seed=3)}
    hot = d["T_C"].to_numpy() >= 215
    splits["hotter_than_seen"] = [(np.where(~hot)[0], np.where(hot)[0])]
    out = {}
    for vn, kw in variants.items():
        F, cats = X(d, **kw)
        out[vn] = {}
        for sn, folds in splits.items():
            p = cv(model, F, cats, y, folds, "point", timer, f"enc_{vn}")
            idx = np.concatenate([te for _, te in folds])
            out[vn][sn] = metrics(y[idx], point=p[idx])
            if sn == "leave_catalyst_out":
                per = {}
                for c in sorted(d["catalyst"].unique()):
                    m = (d["catalyst"] == c).to_numpy()
                    per[c] = {"n": int(m.sum()), "rmse": float(np.sqrt(np.mean((p[m] - y[m]) ** 2)))}
                out[vn][sn]["per_catalyst"] = per
    return out


def exp_missingness(model, timer):
    """Trap 3: promoter blanks that mean 'none used' vs blanks that mean 'not written down'."""
    d, truth = history()
    y = d["yield"].to_numpy(float)
    folds = group_folds(d["batch"], 5)
    out = {}
    kinds = truth["promoter_blank"].to_numpy()
    base = {"catalyst": "PD-A1", "T_C": 200.0, "P_bar": 15.0, "t_h": 4.0, "cat_mol": 3.0, "solvent": "ethanol",
            "stir_rpm": 500.0}
    grid = np.array([0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5.0])
    wr = pd.DataFrame([{**base, "promoter_wt": g} for g in grid])
    wrows, wtruth = fresh(wr, np.random.default_rng(0), observe=False)
    for v in ("mean", "nan", "flag"):
        F, cats = X(d, promoter=v)
        p = cv(model, F, cats, y, folds, "point", timer, f"miss_{v}")
        res = {"all": metrics(y, point=p)}
        for k in ("none used", "not recorded", "recorded"):
            m = kinds == k
            res[k] = metrics(y[m], point=p[m])
        # what-if curve over promoter amount, trained on all 700 rows
        both = pd.concat([d, wrows], ignore_index=True)
        if v == "mean":  # the pipeline fills blanks with the training mean
            mu = d["promoter_wt"].mean()
            both["promoter_wt"] = both["promoter_wt"].fillna(mu)
            Fb, cb = S.features(both, **{**ENGINEERED, "promoter": "nan"})
        else:
            Fb, cb = S.features(both, **{**ENGINEERED, "promoter": v})
        pw = model.point(Fb.iloc[: len(d)], cb, y, Fb.iloc[len(d):])
        res["whatif"] = {"promoter": r(grid, 2), "pred": r(pw), "truth": r(wtruth["true_yield"])}
        out[v] = res
    return out


def logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


def expit(z):
    return 1 / (1 + np.exp(-z))


def exp_bounded(model, timer):
    """Trap 4: raw yield vs logit yield vs two stages (P(success) x yield given success)."""
    d, truth = history()
    y = d["yield"].to_numpy(float)
    F, cats = X(d)
    folds = group_folds(d["batch"], 5)
    zero = y == 0
    out = {"zeros": float(zero.mean()), "failed": float(truth["failed"].mean())}
    raw = cv(model, F, cats, y, folds, "dist", timer, "bounded_raw")
    out["raw"] = metrics(y, raw)
    lg = cv(model, F, cats, logit(y), folds, "dist", timer, "bounded_logit")
    lgy = lg.map(expit)
    out["logit"] = metrics(y, lgy)
    ps = np.clip(cv(model, F, cats, (~zero).astype(float), folds, "point", timer, "bounded_cls"), 0, 1)
    cond = np.full(len(y), np.nan)
    for tr, te in folds:
        tr_ok = tr[~zero[tr]]
        cond[te] = model.point(F.iloc[tr_ok], cats, y[tr_ok], F.iloc[te])
    two = ps * np.clip(cond, 0, 1)
    out["two_stage"] = metrics(y, point=two)
    for k, m in (("raw", raw.mean), ("logit", lgy.mean), ("two_stage", two)):
        out[k]["on_zero_rows"] = float(np.mean(m[zero]))
        out[k]["pred_below0"] = float(np.mean(m < 0))
        out[k]["pred_above1"] = float(np.mean(m > 1))
        out[k]["rmse_vs_truth"] = float(np.sqrt(np.mean((m - truth["true_yield"].to_numpy()) ** 2)))
    lo, hi = raw.interval(0.8)
    out["raw"]["lo80_below0"] = float(np.mean(lo < 0))
    out["rows"] = {"y": r(y, 3), "raw": r(raw.mean, 3), "logit": r(lgy.mean, 3), "two_stage": r(two, 3),
                   "p_success": r(ps, 3), "raw_lo80": r(lo, 3)}
    return out


def exp_selection(model, timer):
    """Trap 6: the same model trained on the biased history vs an unbiased design of the
    same size, both graded on uniform recipes over the design space (T up to 240 C)."""
    rng = np.random.default_rng(21)
    test_rec = S.random_recipes(1500, rng, catalysts=[c.name for c in S.CATALYSTS if c.name not in S.UNSEEN],
                                t_max=S.HIST_T_MAX)
    test, tt = fresh(test_rec, rng)
    metal = test["catalyst"].map(lambda c: S.CAT_BY_NAME[c].metal).to_numpy()
    out = {}
    for name, kw in (("biased", {}), ("unbiased", dict(biased=False))):
        d, _ = history(**kw)
        y = d["yield"].to_numpy(float)
        F, cats = X(pd.concat([d, test], ignore_index=True))
        with timer(f"selection_{name}"):
            dd = model.dist(F.iloc[: len(d)], cats, y, F.iloc[len(d):])
        res = metrics(test["yield"].to_numpy(float), dd)
        truth_y = tt["true_yield"].to_numpy()
        res["bias_vs_truth"] = float(np.mean(dd.mean - truth_y))
        res["rmse_vs_truth"] = float(np.sqrt(np.mean((dd.mean - truth_y) ** 2)))
        res["by_metal"] = {m: {"n": int((metal == m).sum()), "pred": float(dd.mean[metal == m].mean()),
                               "truth": float(truth_y[metal == m].mean()),
                               "observed": float(test["yield"].to_numpy()[metal == m].mean())}
                           for m in S.METALS}
        res["low_yield_region"] = {"pred": float(dd.mean[truth_y < 0.1].mean()),
                                   "truth": float(truth_y[truth_y < 0.1].mean()), "n": int((truth_y < 0.1).sum())}
        out[name] = res
    return out


def exp_causal(model, timer):
    """Trap 7: partial dependence on reaction time vs the true effect of changing time.
    Operators ran poor lots longer, so time and (unrecorded) lot purity are tied."""
    grid = np.exp(np.linspace(np.log(0.5), np.log(40), 12))
    out = {"t_h": r(grid, 3)}
    for name, kw in (("confounded", {}), ("not_confounded", dict(confounded=False))):
        d, truth = history(**kw)
        y = d["yield"].to_numpy(float)
        sub = d.sample(150, random_state=1).reset_index(drop=True)
        subt = truth.loc[d.sample(150, random_state=1).index].reset_index(drop=True)
        rows = pd.concat([sub.assign(t_h=g) for g in grid], ignore_index=True)
        F, cats = X(pd.concat([d, rows], ignore_index=True))
        with timer(f"causal_{name}"):
            p = model.point(F.iloc[: len(d)], cats, y, F.iloc[len(d):]).reshape(len(grid), len(sub))
        rec = rows.copy()
        rec["promoter_wt"] = np.tile(subt["true_promoter"].to_numpy(), len(grid))
        tru = S.true_yield(rec, np.tile(subt["purity"].to_numpy(), len(grid))).reshape(len(grid), len(sub))
        out[name] = {"pd": r(p.mean(axis=1)), "true_effect": r(tru.mean(axis=1)),
                     "corr_lnt_purity": float(np.corrcoef(np.log(d["t_h"]), truth["purity"])[0, 1]),
                     "corr_lnt_yield": float(np.corrcoef(np.log(d["t_h"]), y)[0, 1])}
    return out


def exp_mechanism_probe(model, timer):
    """Expectation 3: the model predicts, but its 'mechanism' must be probed. Query it in
    the low-conversion regime and read an apparent activation energy off ln(y) vs 1/T."""
    d, _ = history()
    y = d["yield"].to_numpy(float)
    probes = (("Pd", "PD-A1", np.arange(165, 216, 5.0)), ("Ni", "NI-A8", np.arange(190, 231, 5.0)))
    out = {}
    rows = []
    for metal, cat, Ts in probes:
        for T in Ts:
            rows.append({"catalyst": cat, "T_C": T, "P_bar": 20.0, "t_h": 2.0, "cat_mol": 5.0, "promoter_wt": 0.0,
                         "solvent": "ethanol", "stir_rpm": 500.0})
    q, qt = fresh(pd.DataFrame(rows), np.random.default_rng(0), observe=False)
    F, cats = X(pd.concat([d, q], ignore_index=True))
    with timer("mechanism_probe"):
        p = model.point(F.iloc[: len(d)], cats, y, F.iloc[len(d):])
    k0 = 0
    for metal, cat, Ts in probes:
        n = len(Ts)
        pm = p[k0:k0 + n]
        tm = qt["true_yield"].to_numpy()[k0:k0 + n]
        k0 += n
        inv = 1 / (Ts + 273.15)
        ok = pm > 0.005
        slope = np.polyfit(inv[ok], np.log(pm[ok]), 1)[0] if ok.sum() >= 3 else np.nan
        tslope = np.polyfit(inv, np.log(np.maximum(tm, 1e-6)), 1)[0]
        out[metal] = {"T": r(Ts, 1), "pred": r(pm), "truth": r(tm), "Ea_apparent": r(-slope * S.R),
                      "Ea_truth_curve": r(-tslope * S.R), "Ea_true": S.METALS[metal][0]}
    return out


def ei_from(d: Dist, best: float) -> np.ndarray:
    return np.mean(np.maximum(d.q - best, 0.0), axis=1)


def exp_acquisition(model, timer, seeds=(0, 1, 2), rounds=10, pool=3000):
    """Expectation 2: use the predictive distribution for expected improvement over a few
    extra experiments. Each new run is one appended row; no retraining."""
    d0, _ = history()
    out = {"seeds": {}}
    for s in seeds:
        rng = np.random.default_rng(100 + s)
        cand_rec = S.random_recipes(pool, rng)
        cand, ct = fresh(cand_rec, rng, observe=False)
        d = d0.copy()
        best = float(d["yield"].max())
        trace = []
        for k in range(rounds):
            F, cats = X(pd.concat([d, cand], ignore_index=True))
            y = d["yield"].to_numpy(float)
            t0 = time.perf_counter()
            if model is None:  # random search baseline
                j = int(rng.integers(len(cand)))
                mu, sd, ei = np.nan, np.nan, np.nan
            else:
                dd = model.dist(F.iloc[: len(d)], cats, y, F.iloc[len(d):])
                e = ei_from(dd, best)
                j = int(np.argmax(e))
                mu, sd, ei = float(dd.mean[j]), float(dd.sd[j]), float(e[j])
            dt = time.perf_counter() - t0
            if timer is not None:
                timer.t.setdefault("acq_update", []).append(dt)
            rec = cand_rec.iloc[[j]].reset_index(drop=True)
            row, rt = fresh(rec, rng)  # run it at the plant
            row["batch"] = f"N{s}{k:02d}"
            row["run_order"] = len(d)
            row["residual_precursor_pct"] = np.nan
            d = pd.concat([d, row], ignore_index=True)
            yobs = float(row["yield"].iloc[0])
            best = max(best, yobs)
            trace.append({"T": float(rec["T_C"].iloc[0]), "catalyst": rec["catalyst"].iloc[0],
                          "t_h": float(rec["t_h"].iloc[0]), "P": float(rec["P_bar"].iloc[0]),
                          "pred": mu, "sd": sd, "ei": ei, "observed": yobs, "true": float(rt["true_yield"].iloc[0]),
                          "failed": bool(rt["failed"].iloc[0]), "seconds": dt})
            cand = cand.drop(index=j).reset_index(drop=True)
            cand_rec = cand_rec.drop(index=cand_rec.index[j]).reset_index(drop=True)
        out["seeds"][str(s)] = trace
    out["hist_best_true_avg_lot"] = float(np.max(S.true_yield(
        d0.assign(promoter_wt=np.where(d0["promoter_used"], d0["promoter_wt"].fillna(2.5), 0.0)), S.AVG_PURITY)))
    out["hist_best_observed"] = float(d0["yield"].max())
    return out


EXPERIMENTS = {"accuracy": exp_accuracy, "extrapolation": exp_extrapolation, "encoding": exp_encoding,
               "missingness": exp_missingness, "bounded": exp_bounded, "selection": exp_selection,
               "causal": exp_causal, "mechanism_probe": exp_mechanism_probe, "acquisition": exp_acquisition}


# ===================================================================== model-free parts
def data_summary() -> dict:
    """Facts about the planted data, and what an ideal model could know (no model needed)."""
    d, truth = history()
    du, tu = history(biased=False)
    rng = np.random.default_rng(5)
    U = S.random_recipes(200000, rng)
    yU = S.true_yield(U, S.AVG_PURITY)
    j = int(np.argmax(yU))
    pf = S.fail_prob(U)
    out = {
        "n": len(d), "batches": int(d["batch"].nunique()),
        "batch_sizes": d["batch"].value_counts().describe().round(2).to_dict(),
        "replicate_rows": int(d.duplicated(subset=S.CONT_RAW + ["catalyst", "solvent"]).sum()),
        "catalyst_counts": d["catalyst"].value_counts().to_dict(),
        "unseen_catalysts": list(S.UNSEEN),
        "promoter_blank": truth["promoter_blank"].value_counts().to_dict(),
        "zeros": float((d["yield"] == 0).mean()), "failed": float(truth["failed"].mean()),
        "yield_hist": np.histogram(d["yield"], bins=20, range=(0, 1))[0].tolist(),
        "yield_hist_unbiased": np.histogram(du["yield"], bins=20, range=(0, 1))[0].tolist(),
        "design_true_hist": np.histogram(yU, bins=20, range=(0, 1))[0].tolist(),
        "T_hist": np.histogram(d["T_C"], bins=20, range=(100, 300))[0].tolist(),
        "T_hist_unbiased": np.histogram(du["T_C"], bins=20, range=(100, 300))[0].tolist(),
        "metal_share": d["catalyst"].map(lambda c: S.CAT_BY_NAME[c].metal).value_counts(normalize=True).round(3).to_dict(),
        "metal_share_unbiased": du["catalyst"].map(lambda c: S.CAT_BY_NAME[c].metal).value_counts(normalize=True).round(3).to_dict(),
        "optimum": {**{k: (v if isinstance(v, str) else round(float(v), 2)) for k, v in U.iloc[j].to_dict().items()},
                    "true_yield": float(yU[j])},
        "hist_best_true_avg_lot": float(np.max(S.true_yield(d.assign(promoter_wt=truth["true_promoter"]), S.AVG_PURITY))),
        "fail_curve": {"T": list(range(200, 301, 10)),
                       "p": [float(S.fail_prob(pd.DataFrame({"T_C": [T], "P_bar": [15.0]}))[0]) for T in range(200, 301, 10)]},
        "corr_lnt_purity": float(np.corrcoef(np.log(d["t_h"]), truth["purity"])[0, 1]),
        "batch_effect_share": float(_batch_share(d, truth)),
        "scatter_T_t": {"T": r(d["T_C"], 1), "t": r(d["t_h"], 2), "purity": r(truth["purity"], 3), "y": r(d["yield"], 3)},
    }
    return out


def _batch_share(d, truth):
    """Share of yield variance explained by the batch (lot) after the recipe: true yield at
    the batch's purity minus at the average purity."""
    rec = d.assign(promoter_wt=truth["true_promoter"])
    diff = S.true_yield(rec, truth["purity"].to_numpy()) - S.true_yield(rec, S.AVG_PURITY)
    return np.var(diff) / np.var(d["yield"])


def mechanism_sparse() -> dict:
    """Expectation 3: recover the mechanism with statistics, not with the foundation model.
    Initial-rate regime (yield < 0.3, below 220 C, product not yet over-reacting):
    ln y ~ ln k1 + ln t, so a sparse fit on physical terms should give Ea per metal."""
    from sklearn.linear_model import LassoCV, LinearRegression

    out = {}
    for name, kw in (("history", {}), ("clean_design", dict(biased=False, confounded=False))):
        d, truth = history(**kw)
        m = (d["yield"] > 0.03) & (d["yield"] < 0.3) & (d["T_C"] < 220)
        dd = d[m].reset_index(drop=True)
        tp = truth.loc[m, "true_promoter"].to_numpy()
        metal = dd["catalyst"].map(lambda c: S.CAT_BY_NAME[c].metal)
        cat = dd["catalyst"].map(S.CAT_BY_NAME)
        lib = pd.DataFrame({"ln_t": np.log(dd["t_h"]), "t": dd["t_h"], "T": dd["T_C"], "T2": (dd["T_C"] / 100) ** 2,
                            "ln_P": np.log(dd["P_bar"]), "P": dd["P_bar"], "ln_cat_frac": np.log(dd["cat_mol"] / (dd["cat_mol"] + 1.5)),
                            "cat_mol": dd["cat_mol"], "promoter_term": np.log1p(0.8 * tp / (tp + 0.7)),
                            "ln_loading": np.log(cat.map(lambda c: c.loading)), "ox": cat.map(lambda c: float(c.ox_state > 0)),
                            "ln_bet": np.log(cat.map(lambda c: S.SUPPORTS[c.support][0])), "stir": dd["stir_rpm"] / 1000,
                            **{f"solvent_{s}": (dd["solvent"] == s).astype(float) for s in S.SOLVENTS}})
        for mt in S.METALS:
            lib[f"metal_{mt}"] = (metal == mt).astype(float)
            lib[f"invT_{mt}"] = (metal == mt) * (1 / (S.R * (dd["T_C"] + 273.15)))
        cols = list(lib.columns)
        Z = (lib - lib.mean()) / lib.std().replace(0, 1)
        yv = np.log(dd["yield"].to_numpy())
        las = LassoCV(cv=5, random_state=0, max_iter=50000).fit(Z.fillna(0), yv)
        keep = [c for c, b in zip(cols, las.coef_) if abs(b) > 1e-6]
        ols = LinearRegression().fit(lib[keep], yv)
        coef = dict(zip(keep, ols.coef_))
        ea = {mt: {"est": float(-coef[f"invT_{mt}"]) if f"invT_{mt}" in coef else None,
                   "true": S.METALS[mt][0] + 0.0, "n": int((metal == mt).sum())} for mt in S.METALS}
        out[name] = {"rows_used": int(m.sum()), "selected": keep, "dropped": [c for c in cols if c not in keep],
                     "coef": {k: float(v) for k, v in coef.items()}, "Ea": ea,
                     "ln_t_coef": float(coef.get("ln_t", np.nan)), "stir_selected": "stir" in keep}
    # GP with ARD on the engineered features: which inputs get short length scales?
    from .models import GPModel

    d, _ = history()
    F, cats = X(d)
    gp = GPModel(maxiter=300)
    gp.point(F, cats, d["yield"].to_numpy(float), F.iloc[:5])
    out["gp_ard_lengthscales"] = gp.last_lengthscales
    return out


def licenses() -> list[dict]:
    from ..models import model_info

    rows = model_info()
    rows.append({"name": "gbdt", "license": "BSD-3-Clause (scikit-learn)", "commercial_use": True, "install": "included"})
    rows += [{"name": "tabpfn-v2", "license": "Prior Labs License (Apache-2.0 + attribution clause)",
              "commercial_use": None, "install": "not used here"},
             {"name": "tabicl", "license": "BSD-3-Clause", "commercial_use": True, "install": "not installed"}]
    return rows


# ======================================================================== command line
def run_model(name: str, experiments: list[str], fast: bool = False) -> dict:
    timer = Timer()
    model = None if name == "random" else build(name, fast=fast)
    res = {"model": name, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": platform.node(),
           "python": sys.version.split()[0], "experiments": {}, "errors": {}}
    try:
        import torch

        res["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        res["device"] = "cpu"
    for e in experiments:
        if name == "random" and e != "acquisition":
            continue
        t0 = time.perf_counter()
        print(f"[{name}] {e} ...", flush=True)
        try:
            res["experiments"][e] = EXPERIMENTS[e](model, timer)
        except Exception as ex:  # keep going; the report says what failed
            res["errors"][e] = traceback.format_exc()[-2000:]
            print(f"[{name}] {e} FAILED: {ex}", flush=True)
        print(f"[{name}] {e} done in {time.perf_counter() - t0:.0f}s", flush=True)
    res["timing"] = {k: {"n": len(v), "median_s": float(np.median(v)), "total_s": float(np.sum(v))}
                     for k, v in timer.t.items()}
    if model is not None:
        res["dist_source"] = getattr(model, "dist_source", None)
        res["native_error"] = getattr(model, "native_error", None)
        res["version"] = getattr(getattr(model, "cls", None), "label", name)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["gp", "gbdt"])
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS))
    ap.add_argument("--out", default="results/traps")
    ap.add_argument("--fast", action="store_true", help="fewer bootstrap passes for in-context models")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if "none" in a.models or "data" in a.models:
        (out / "data.json").write_text(json.dumps({"summary": data_summary(), "mechanism": mechanism_sparse(),
                                                   "licenses": licenses()}, default=str))
        print("wrote", out / "data.json")
    for m in a.models:
        if m in ("none", "data"):
            continue
        res = run_model(m, a.experiments, fast=a.fast)
        p = out / f"{m}.json"
        p.write_text(json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        print("wrote", p, "errors:", list(res["errors"]))


if __name__ == "__main__":
    main()
