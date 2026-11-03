"""Case study on the fake plant: every screen, every model, compared with the known truth.

The fake plant's yield formula is known, so each result can be checked: which findings are
real, how far each curve is from the true one, how good the suggested recipes really are.
It also plants the data problems a real plant has without knowing it:

* clean:  the history as recorded (it already has aliasing, drift and a no-data region);
* typos:  five hand-entry slips, three of which stay inside the allowed ranges;
* leak:   the measured-during-run column (최고 온도) mislabelled as a setting;

and runs a short campaign (rounds of suggested batches, with the default limit of two
changed conditions per run and with no limit) to see how close each model gets to the
true best recipe. Everything goes into one JSON file for the report page.

    python -m yieldengine.casestudy --models gp tabfm limix-16m exaone-tabular \
        --device cuda --out results/casestudy.json
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis.importance import Group, find_groups, grouped_importance
from .calibration import time_order
from .cleaning import flag_suspect_values
from .engine import Engine
from .schema import Role, Schema, Variable
from .simulator import FakePlant, default_schema
from .suggest import BatchSettings

CONTROLS = ["temp", "pressure", "gas_flow", "time", "stir", "catalyst"]

def _r(x, d=2):
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return [_r(v, d) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), d)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _recipe(row: dict) -> dict:
    return {k: (str(row[k]) if k == "catalyst" else _r(row[k])) for k in CONTROLS}


def truth_optimum(plant: FakePlant, context: dict) -> dict:
    T, G, M, C = np.meshgrid(np.arange(150, 201, 1.0), np.arange(10, 30.01, 0.5), np.arange(60, 181, 5.0),
                             np.array(["A", "B", "C"], dtype=object), indexing="ij")
    grid = pd.DataFrame({"temp": T.ravel(), "gas_flow": G.ravel(), "time": M.ravel(), "catalyst": C.ravel(),
                         "pressure": 2.0, "stir": 400.0, **context})
    y = plant.true_yield(grid)
    i = int(np.argmax(y))
    return {"recipe": _recipe(grid.iloc[i].to_dict()), "yield": _r(y[i])}


def make_typos(history: pd.DataFrame, schema: Schema) -> tuple[pd.DataFrame, list[dict]]:
    """Five slips a person typing up run sheets could make. Returns the data and what changed."""
    d = history.copy()
    runs = d[~d["is_reference"].astype(bool)]
    log = []

    def put(i, col, new, kind):
        log.append({"row": int(i), "days": _r(d.at[i, "days"]), "column": col, "true_value": _r(d.at[i, col]),
                    "typed": _r(new), "kind": kind, "catalyst": str(d.at[i, "catalyst"])})
        d.at[i, col] = new

    b = runs[runs["catalyst"] == "B"].sort_values("yield", ascending=False)
    i = b.index[0]  # the best B run: digits swapped
    s = f"{d.at[i, 'yield']:.1f}"
    put(i, "yield", float(s[1] + s[0] + s[2:]), "자리 바뀜 (digits swapped)")
    a = runs[(runs["catalyst"] == "A") & (runs["days"] < 120)]
    i = a.index[len(a) // 2]
    put(i, "yield", round(float(d.at[i, "yield"]) / 10, 2), "소수점 (decimal slip)")
    bt = runs[(runs["catalyst"] == "B") & (runs["temp"] >= 176)]
    i = bt.index[len(bt) // 3]
    t = int(d.at[i, "temp"])
    swapped = int(str(t)[0] + str(t)[2] + str(t)[1])
    put(i, "temp", float(swapped if 150 <= swapped <= 200 else t - 9), "자리 바뀜, 범위 안 (in range)")
    g = runs[runs["gas_flow"].map(lambda v: 21 <= v < 30 and str(int(v))[0] != str(int(v))[1])]
    i = g.index[len(g) // 2]
    v = float(d.at[i, "gas_flow"])
    typed = float(str(int(v))[::-1]) + (v % 1)
    put(i, "gas_flow", typed, "자리 바뀜, 범위 안 (in range)" if 10 <= typed <= 30 else "자리 바뀜, 범위 밖 (out of range)")
    m = runs[runs["time"] == 150]
    i = m.index[len(m) // 2]
    put(i, "time", 15.0, "0 빠짐 (dropped zero)")
    return d, log


def leak_schema(schema: Schema) -> Schema:
    """최고 온도 (measured during the run) mislabelled as a setting."""
    new = []
    for v in schema.variables:
        if v.name == "peak_temp":
            v = Variable(**{**v.__dict__, "role": Role.SET_BEFORE, "standard": 173.0})
        new.append(v)
    return Schema(new, schema.date_column, schema.reference_column)


PROBES = {
    "standard": {},
    "B_185": {"catalyst": "B", "temp": 185},
    "B_195": {"catalyst": "B", "temp": 195},
    "B_185_high_gas": {"catalyst": "B", "temp": 185, "gas_flow": 28, "pressure": 2.8, "time": 150},
    "A_gas_up_pressure_down": {"gas_flow": 28, "pressure": 1.2},
    "A_pressure_up_gas_down": {"gas_flow": 12, "pressure": 2.8},
    "C_180": {"catalyst": "C", "temp": 180},
}


def truth_importance(plant: FakePlant, data: pd.DataFrame, schema: Schema) -> dict:
    """The same permutation measure the engine uses, applied to the true formula."""

    class Truth:
        def predict(self, rows):
            from .models import Prediction

            m = plant.true_yield(rows)
            return Prediction(m, np.ones_like(m))

    groups = find_groups(data, schema)
    per_group = grouped_importance(Truth(), data, groups)
    per_var = grouped_importance(Truth(), data, [Group([v.name]) for v in schema.features])
    out = lambda df: [{"group": r.group, "importance": _r(r.importance), "share": _r(r.share, 3)}
                      for r in df.itertuples()]
    return {"groups": out(per_group), "variables": out(per_var)}


def truth_curves(plant: FakePlant, schema: Schema, context: dict, n_grid: int = 26) -> dict:
    out = {}
    std = schema.standard_recipe()
    for base_name, base in (("B", {**std, "catalyst": "B"}), ("A", std)):
        for v in schema.controls:
            if not v.is_continuous:
                continue
            xs = np.linspace(v.lower, v.upper, n_grid)
            rows = pd.DataFrame([{**base, **context, v.name: x} for x in xs])
            out[f"{base_name}:{v.name}"] = {"value": _r(xs), "true": _r(plant.true_yield(rows))}
    return out


def run_engine(name: str, data: pd.DataFrame, schema: Schema, plant: FakePlant, context: dict,
               kw: dict, findings: bool = True, batch_seed: int = 0) -> dict:
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = Engine(data, schema, model=name, context=context, **kw)
        model = eng.model
    out = {"model": eng.model_name, "fallback": eng.fallback_reason, "factor": _r(model.factor, 3)}
    t1 = time.time()

    # out-of-time accuracy (rolling origin)
    e = eng.calibration_errors.copy()
    d = time_order(eng.data, schema).reset_index(drop=True)
    e["days"] = d["days"].to_numpy()[e["row"].to_numpy()]
    e["true"] = plant.true_yield(d.iloc[e["row"].to_numpy()])
    lo, hi = e["mean"] - 1.2816 * e["sd"] * model.factor, e["mean"] + 1.2816 * e["sd"] * model.factor
    resid = e["actual"] - e["mean"]
    out["oot"] = {
        "days": _r(e["days"], 0), "actual": _r(e["actual"]), "true": _r(e["true"]), "mean": _r(e["mean"]),
        "low": _r(lo), "high": _r(hi),
        "rmse": _r(np.sqrt(np.mean(resid**2))),
        "r2": _r(1 - np.sum(resid**2) / np.sum((e["actual"] - e["actual"].mean()) ** 2), 3),
        "coverage_raw": _r(np.mean(np.abs(e["z"]) <= 1.2816), 3),
        "coverage": _r(np.mean((e["actual"] >= lo) & (e["actual"] <= hi)), 3),
    }
    out["drift"] = None if eng.drift is None else {"delta": _r(eng.drift.delta), "se": _r(eng.drift.se),
                                                  "applied": eng.drift.applied, "note": eng.drift.note}
    # importance of the final model on the data, same groups as the findings
    groups = find_groups(eng.data, schema)
    imp = grouped_importance(model, eng.data, groups)
    out["importance"] = [{"group": r.group, "importance": _r(r.importance), "share": _r(r.share, 3)}
                         for r in imp.itertuples()]
    t2 = time.time()
    if findings:
        out["findings"] = [{"kind": f.kind, "strength": f.strength, "sentence": f.sentence, "key": f.key,
                            "facts": {k: _r(v) for k, v in f.facts.items() if not isinstance(v, (dict, list))}}
                           for f in eng.findings(include_weak=True)]
    t3 = time.time()

    # 바꿔보기: curves around catalyst B and A, and the probe recipes
    std = schema.standard_recipe()
    curves = {}
    for base_name, base in (("B", {**std, "catalyst": "B"}), ("A", std)):
        for vname, c in eng.slider_curves(base).items():
            if not schema[vname].is_continuous:
                curves[f"{base_name}:{vname}"] = {"value": [str(x) for x in c["value"]], "mean": _r(c["mean"]),
                                                  "low": _r(c["low"]), "high": _r(c["high"]),
                                                  "n_nearby": _r(c["n_nearby"])}
                continue
            curves[f"{base_name}:{vname}"] = {"value": _r(c["value"]), "mean": _r(c["mean"]), "low": _r(c["low"]),
                                              "high": _r(c["high"]), "supported": _r(c["supported"])}
    out["curves"] = curves
    probes = {}
    for k, change in PROBES.items():
        r = {**std, **change}
        if "peak_temp" in r:  # leak schema: a person would type the usual peak
            r["peak_temp"] = r["temp"] + 3 + (2 if r["catalyst"] == "B" else 0)
        p = eng.predict(r)
        probes[k] = {**p, "true": _r(plant.true_yield(pd.DataFrame([{**r, **context}]))[0])}
    out["probes"] = probes

    # 다음 실험: the default batch, with the true yield of every suggestion
    batch = eng.suggest(BatchSettings(n_runs=12, seed=batch_seed))
    frame = batch.to_frame()
    truth = plant.true_yield(frame)
    out["batch"] = {"notes": batch.notes, "runs": [
        {"run": s.run, "purpose": s.purpose, "recipe": _recipe(s.recipe), "predicted": _r(s.predicted),
         "low": _r(s.low), "high": _r(s.high), "true": _r(t), "reason": s.reason,
         "changes": [[a, _r(b), _r(c)] for a, b, c in s.changes],
         "new_conditions": s.facts.get("new_conditions")} for s, t in zip(batch.suggestions, truth)]}
    if "peak_temp" in frame:
        out["batch"]["peak_temp_set"] = _r(frame["peak_temp"])
    t4 = time.time()
    out["seconds"] = {"fit": _r(t1 - t0, 1), "importance": _r(t2 - t1, 1), "findings": _r(t3 - t2, 1),
                      "whatif_batch": _r(t4 - t3, 1)}
    return out


def campaign(name: str, schema: Schema, seed: int, rounds: int, kw: dict, max_changes: int = 2) -> dict:
    """Start from the recorded history; each round, run the suggested batch on the plant."""
    plant = FakePlant(seed=0)
    data = plant.history()
    runner = FakePlant(seed=100 + seed)  # noise for the new runs
    runner.lot_effect = plant.lot_effect
    best_true, best_seen, ctx_days, best_recipe, best_value = [], [], float(data["days"].max()), None, -np.inf
    for r in range(rounds):
        ctx = {"lot": data.sort_values("days")["lot"].iloc[-1], "days": ctx_days + 1 + 7 * r}
        if name == "random":
            rng = np.random.default_rng(seed * 100 + r)
            rows = pd.DataFrame({"temp": rng.integers(150, 201, 12).astype(float),
                                 "pressure": np.round(rng.uniform(1, 3, 12), 1),
                                 "gas_flow": np.round(rng.uniform(10, 30, 12) * 2) / 2,
                                 "time": rng.choice(np.arange(60, 181, 5), 12).astype(float),
                                 "stir": np.round(rng.uniform(200, 600, 12), -1),
                                 "catalyst": rng.choice(["A", "B", "C"], 12), **ctx})
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eng = Engine(data, schema, model=name, context=ctx, **kw)
                rows = eng.suggest(BatchSettings(n_runs=12, seed=seed + r, max_changes=max_changes)).to_frame()
            rows = rows[CONTROLS + ["lot", "days"]]
        rows["is_reference"] = False
        rows["date"] = pd.Timestamp("2026-03-02") + pd.to_timedelta(rows["days"], unit="D")
        ran = runner.run(rows)
        truth = plant.true_yield(ran)
        best_true.append(_r(float(truth.max())))
        best_seen.append(_r(float(ran["yield"].max())))
        if truth.max() > best_value:
            best_value, best_recipe = float(truth.max()), _recipe(ran.iloc[int(np.argmax(truth))].to_dict())
        data = pd.concat([data, ran], ignore_index=True)
    return {"best_true_per_round": best_true, "best_measured_per_round": best_seen, "best_recipe": best_recipe}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gp"])
    ap.add_argument("--scenarios", nargs="+", default=["clean", "typos", "leak", "nodrift"])
    ap.add_argument("--campaign-rounds", type=int, default=3)
    ap.add_argument("--campaign-seeds", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/casestudy.json")
    a = ap.parse_args(argv)

    schema = default_schema()
    plant = FakePlant(seed=0)
    history = plant.history()
    context = plant.context_now(history)
    typo_data, typo_log = make_typos(history, schema)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}

    hist = history.copy()
    hist["true"] = plant.true_yield(hist)
    res["data"] = {
        "context": {k: _r(v) for k, v in context.items()},
        "optimum": truth_optimum(plant, context),
        "runs": [{k: (str(v) if isinstance(v, str) else _r(v)) for k, v in row.items() if k != "date"}
                 for row in hist.to_dict("records")],
        "truth_importance": truth_importance(plant, history, schema),
        "truth_curves": truth_curves(plant, schema, context),
        "typos": typo_log,
        "flags_clean": flag_suspect_values(history, schema).astype(object).to_dict("records"),
        "flags_typos": flag_suspect_values(typo_data, schema).astype(object).to_dict("records"),
        "lot_effect": {k: _r(v) for k, v in plant.lot_effect.items()},
    }
    res.setdefault("models", {})
    res.setdefault("campaign", {})

    def save():
        out_path.write_text(json.dumps(res, ensure_ascii=False, default=lambda o: _r(o) if not isinstance(o, str) else o))

    save()
    for name in a.models:
        kw = {"device": a.device} if a.device and name != "gp" else {}
        m = res["models"].setdefault(name, {})
        for sc in a.scenarios:
            t = time.time()
            if sc == "clean":
                m["clean"] = run_engine(name, history, schema, plant, context, kw)
            elif sc == "typos":
                m["typos"] = run_engine(name, typo_data, schema, plant, context, kw)
            elif sc == "leak":
                m["leak"] = run_engine(name, history, leak_schema(schema), plant, context, kw)
            elif sc == "nodrift":
                m["nodrift"] = run_engine(name, history, schema, plant, context, {**kw, "drift_correction": False},
                                          findings=False)
            print(f"{name} {sc}: {time.time() - t:.0f} s", flush=True)
            save()
        if a.campaign_rounds:
            t = time.time()
            res["campaign"][name] = {
                "max2": [campaign(name, schema, s, a.campaign_rounds, kw, 2) for s in a.campaign_seeds],
                "max6": [campaign(name, schema, s, a.campaign_rounds, kw, 6) for s in a.campaign_seeds]}
            print(f"{name} campaign: {time.time() - t:.0f} s", flush=True)
            save()
    if a.campaign_rounds and "random" not in res["campaign"]:
        res["campaign"]["random"] = {"any": [campaign("random", schema, s, a.campaign_rounds, {})
                                             for s in a.campaign_seeds]}
        save()
    print(f"wrote {out_path} ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
