"""A fake catalytic synthesis with a known true mechanism, for the traps case study.

Every trap from the "what to expect from a tabular foundation model" answer is planted
here on purpose, so each one can be checked against the truth:

* Arrhenius kinetics in series, A -> B (catalysed, Ea by metal) -> C (thermal, Ea 130),
  so the best time and temperature are inside the range, plus a support breakdown above
  ~245 C that cuts selectivity. Ea depends on the metal, so
  catalyst descriptors (metal, oxidation state, loading, support area) carry the
  chemistry while the catalyst *name* carries none of it (trap 2).
* Promoter: blank when no promoter was used (structured missingness) and also blank
  when it was used but not written down (random missingness) (trap 3).
* Yield is in [0, 1]; failed runs are written down as 0 and yields under a 3 % detection
  limit read 0 (trap 4).
* Runs come in batches that share one precursor lot. Lot purity changes the rate and is
  never recorded: a hidden group effect (trap 5). A measured column (residual precursor)
  is recorded after the run and encodes the answer (trap 5, leakage).
* The history over-represents what chemists expected to work (Pd/Pt, 170-230 C) and
  drops many low-yield runs (trap 6).
* Operators ran longer when a lot looked poor, so in the history long times go with bad
  lots. The true effect of time is positive; the recorded association is weaker (trap 7).
* Historical temperatures stop at 240 C while the design space goes to 300 C (trap 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

R = 8.314e-3  # kJ/(mol K)
EA2, LN_A2 = 130.0, 30.28  # over-reaction of the product (A -> B -> C)

# metal -> activation energy (kJ/mol) and log pre-exponential offset
METALS = {"Pd": (62.0, 0.0), "Pt": (66.0, 0.6), "Ni": (78.0, 2.4), "Co": (82.0, 3.0), "Cu": (88.0, 3.9), "Fe": (96.0, 5.4)}
SUPPORTS = {"Al2O3": (180.0, 0.0), "SiO2": (300.0, -10.0), "C": (900.0, -15.0), "TiO2": (55.0, 12.0)}  # BET m2/g, Tdec shift
SOLVENTS = {"water": 0.85, "ethanol": 1.0, "toluene": 0.75, "DMF": 1.15}


@dataclass(frozen=True)
class Catalyst:
    name: str
    metal: str
    support: str
    loading: float  # wt%
    ox_state: int


CATALYSTS = [
    Catalyst("PD-A1", "Pd", "Al2O3", 5.0, 0), Catalyst("PD-C2", "Pd", "C", 10.0, 0),
    Catalyst("PD-S3", "Pd", "SiO2", 2.0, 2), Catalyst("PD-T4", "Pd", "TiO2", 1.0, 0),
    Catalyst("PT-A5", "Pt", "Al2O3", 1.0, 0), Catalyst("PT-C6", "Pt", "C", 5.0, 0),
    Catalyst("PT-S7", "Pt", "SiO2", 3.0, 2),
    Catalyst("NI-A8", "Ni", "Al2O3", 10.0, 0), Catalyst("NI-S9", "Ni", "SiO2", 20.0, 2),
    Catalyst("NI-T10", "Ni", "TiO2", 8.0, 0),
    Catalyst("CO-A11", "Co", "Al2O3", 15.0, 0), Catalyst("CO-C12", "Co", "C", 10.0, 2),
    Catalyst("CU-S13", "Cu", "SiO2", 10.0, 2), Catalyst("CU-A14", "Cu", "Al2O3", 15.0, 2),
    Catalyst("FE-A15", "Fe", "Al2O3", 20.0, 3), Catalyst("FE-C16", "Fe", "C", 15.0, 3),
]
CAT_BY_NAME = {c.name: c for c in CATALYSTS}
# never tried in the history; still in the design space
UNSEEN = ("NI-T10", "PT-S7")

BOUNDS = {"T_C": (100.0, 300.0), "P_bar": (1.0, 60.0), "t_h": (0.25, 48.0), "cat_mol": (0.5, 10.0),
          "promoter_wt": (0.0, 5.0), "stir_rpm": (200.0, 1200.0)}
HIST_T_MAX = 240.0
AVG_PURITY = 0.86  # mean of the lot purity draw U(0.72, 1)


def descriptors(name: str) -> dict:
    c = CAT_BY_NAME[name]
    return {"metal": c.metal, "support": c.support, "loading_wt": c.loading, "ox_state": c.ox_state,
            "bet_m2g": SUPPORTS[c.support][0]}


# ----------------------------------------------------------------------------- truth
def true_yield(df: pd.DataFrame, purity: np.ndarray | float = 0.86) -> np.ndarray:
    """Noise-free yield in [0, 1] for recipe rows. ``promoter_wt`` must be the TRUE
    amount (0 when none was used). ``purity`` is the unrecorded precursor-lot purity."""
    T = df["T_C"].to_numpy(float) + 273.15
    t = df["t_h"].to_numpy(float)
    P = df["P_bar"].to_numpy(float)
    cm = df["cat_mol"].to_numpy(float)
    pr = df["promoter_wt"].to_numpy(float)
    cats = [CAT_BY_NAME[c] for c in df["catalyst"]]
    Ea = np.array([METALS[c.metal][0] + 5.0 * (c.ox_state > 0) for c in cats])
    lnA = np.array([METALS[c.metal][1] + 0.35 * np.log(c.loading) + 0.25 * np.log(SUPPORTS[c.support][0] / 180.0)
                    for c in cats])
    tdec = np.array([248.0 + SUPPORTS[c.support][1] for c in cats])
    solvent = df["solvent"].map(SOLVENTS).to_numpy(float)
    purity = np.broadcast_to(np.asarray(purity, float), T.shape)
    lnk = (13.4 + lnA - Ea / (R * T) + np.log(cm / (cm + 1.5)) + 0.4 * np.log(P / 10.0)
           + np.log1p(0.8 * pr / (pr + 0.7)) + np.log(solvent) + 3.0 * np.log(purity))
    k1 = np.exp(lnk)
    k2 = np.exp(LN_A2 - EA2 / (R * T))  # product -> by-product, thermal, catalyst-independent
    d = k2 - k1
    safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
    formed = np.where(np.abs(d) < 1e-9, k1 * t * np.exp(-k1 * t), k1 / safe * (np.exp(-k1 * t) - np.exp(-k2 * t)))
    sel = 1.0 / (1.0 + np.exp((df["T_C"].to_numpy(float) - tdec) / 9.0))  # support breaks down
    return np.clip(formed, 0, 1) * sel


def fail_prob(df: pd.DataFrame) -> np.ndarray:
    """Runs fail (gel, runaway, reactor trip) more often when hot and pressurised."""
    T = df["T_C"].to_numpy(float)
    P = df["P_bar"].to_numpy(float)
    return 0.02 + 0.5 / (1.0 + np.exp(-(T - 262.0) / 8.0)) + 0.10 / (1.0 + np.exp(-(P - 50.0) / 3.0))


def observe(df: pd.DataFrame, purity, rng: np.random.Generator, noise_sd: float = 0.025):
    """Measured yield, with failures written down as 0 and a 3 % detection limit."""
    y = true_yield(df, purity) + rng.normal(0, noise_sd, len(df))
    failed = rng.random(len(df)) < fail_prob(df)
    y = np.where(failed, 0.0, y)
    y = np.where(y < 0.03, 0.0, y)
    return np.clip(y, 0.0, 1.0), failed


# ----------------------------------------------------------------------------- sampling
def random_recipes(n: int, rng: np.random.Generator, catalysts=None, t_max: float = BOUNDS["T_C"][1]) -> pd.DataFrame:
    """Uniform draws over the design space (log-uniform for time, pressure, amounts)."""
    names = [c.name for c in CATALYSTS] if catalysts is None else list(catalysts)
    lo = lambda k: BOUNDS[k][0]
    hi = lambda k: BOUNDS[k][1]
    used = rng.random(n) < 0.6
    df = pd.DataFrame({
        "catalyst": rng.choice(names, n),
        "T_C": rng.uniform(lo("T_C"), t_max, n),
        "P_bar": np.exp(rng.uniform(np.log(lo("P_bar")), np.log(hi("P_bar")), n)),
        "t_h": np.exp(rng.uniform(np.log(lo("t_h")), np.log(hi("t_h")), n)),
        "cat_mol": np.exp(rng.uniform(np.log(lo("cat_mol")), np.log(hi("cat_mol")), n)),
        "promoter_wt": np.where(used, rng.uniform(0.2, 5.0, n), 0.0),
        "solvent": rng.choice(list(SOLVENTS), n),
        "stir_rpm": rng.uniform(lo("stir_rpm"), hi("stir_rpm"), n),
    })
    return df


@dataclass
class History:
    """The 700-row plant history plus everything the case study needs to grade it."""

    data: pd.DataFrame  # what the engineers see (blanks, zeros, measured column, batch ids)
    truth: pd.DataFrame  # true promoter, purity, noise-free yield, failed flag per row
    meta: dict = field(default_factory=dict)


CAT_POPULARITY = {"Pd": 6.0, "Pt": 3.5, "Ni": 1.6, "Co": 1.0, "Cu": 0.45, "Fe": 0.3}


def make_history(n_rows: int = 700, seed: int = 0, biased: bool = True, confounded: bool = True) -> History:
    """Batches of runs on one precursor lot each, in time order.

    ``biased``: chemists pick popular catalysts and 'sensible' temperatures, and many
    low-yield runs never make it into the table. ``confounded``: operators run longer on
    lots that look poor. With both False the history is a uniform design with the same
    batch structure (the counterfactual dataset only a simulation can give you)."""
    rng = np.random.default_rng(seed)
    seen = [c for c in CATALYSTS if c.name not in UNSEEN]
    w = np.array([CAT_POPULARITY[c.metal] if biased else 1.0 for c in seen])
    w = w / w.sum()
    rows, truth = [], []
    batch = 0
    kept = 0
    while kept < n_rows:
        batch += 1
        purity = rng.uniform(0.72, 1.0)
        lab = "A" if rng.random() < 0.65 else "B"
        size = int(rng.integers(5, 13))
        cat = seen[rng.choice(len(seen), p=w)].name
        if biased:
            T0 = float(np.clip(rng.normal(200, 18), 130, 236))
            lt0 = rng.normal(np.log(4.0), 0.7)
        else:
            T0 = rng.uniform(110, 236)
            lt0 = rng.uniform(np.log(0.4), np.log(40))
        if confounded:  # poor lots get longer runs ("it needs more time")
            lt0 = lt0 + 3.2 * (0.86 - purity)
        P0 = np.exp(rng.uniform(np.log(2), np.log(40)))
        cm0 = np.exp(rng.uniform(np.log(0.7), np.log(8)))
        promo_used = rng.random() < 0.55
        pr0 = rng.uniform(0.3, 4.5) if promo_used else 0.0
        solvent = rng.choice(list(SOLVENTS), p=[0.15, 0.45, 0.15, 0.25] if biased else None)
        prev = None
        for i in range(size):
            if prev is not None and rng.random() < 0.2:  # replicate of the previous run
                r = dict(prev)
            else:
                r = {"catalyst": cat, "T_C": float(np.clip(T0 + rng.normal(0, 9), 110, HIST_T_MAX)),
                     "P_bar": float(np.clip(P0 * np.exp(rng.normal(0, 0.3)), 1, 50)),
                     "t_h": float(np.clip(np.exp(lt0 + rng.normal(0, 0.45)), 0.25, 40)),
                     "cat_mol": float(np.clip(cm0 * np.exp(rng.normal(0, 0.25)), 0.5, 10)),
                     "promoter_wt": float(np.clip(pr0 * np.exp(rng.normal(0, 0.2)), 0.2, 5)) if promo_used else 0.0,
                     "solvent": solvent if rng.random() < 0.85 else rng.choice(list(SOLVENTS)),
                     "stir_rpm": float(rng.choice([300, 500, 800]) if biased else rng.uniform(200, 1200))}
            prev = r
            one = pd.DataFrame([r])
            y, failed = observe(one, purity, rng)
            yt = float(true_yield(one, purity)[0])
            if biased and y[0] < 0.25 and rng.random() < 0.6:  # never written up
                continue
            rows.append({**r, "batch": f"B{batch:03d}", "lab": lab, "yield": float(y[0])})
            truth.append({"purity": purity, "true_promoter": r["promoter_wt"], "true_yield": yt,
                          "failed": bool(failed[0])})
            kept += 1
            if kept >= n_rows:
                break
    data = pd.DataFrame(rows)
    truth = pd.DataFrame(truth)
    n = len(data)
    data["run_order"] = np.arange(n)
    # measured after the run: unreacted precursor, read off the same assay as the yield
    data["residual_precursor_pct"] = np.clip(
        100 * (1 - data["yield"]) * rng.uniform(0.85, 1.0, n) + rng.normal(0, 2, n), 0, 100)
    # structured blank: no promoter used -> blank; random blank: used but not written down
    blank_random = (truth["true_promoter"] > 0) & (rng.random(n) < 0.12)
    data["promoter_wt"] = np.where((truth["true_promoter"] == 0) | blank_random, np.nan, truth["true_promoter"])
    truth["promoter_blank"] = np.where(truth["true_promoter"] == 0, "none used",
                                       np.where(blank_random, "not recorded", "recorded"))
    meta = {"batches": int(data["batch"].nunique()), "seed": seed, "biased": biased, "confounded": confounded}
    return History(data, truth, meta)


# ----------------------------------------------------------------------------- features
CONT_RAW = ["T_C", "P_bar", "t_h", "cat_mol", "stir_rpm"]


def features(df: pd.DataFrame, *, labels: bool = True, desc: bool = True, transformed: bool = True,
             promoter: str = "flag", leak: bool = False, lab: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Model inputs for one encoding choice. Returns (frame, categorical column names).

    promoter: "flag" = blank-because-none -> 0 plus a used flag (needs ``promoter_used``
    in df, i.e. the recipe sheet); "nan" = leave blanks as missing; "mean" = fill blanks
    with the column mean (the usual pipeline default)."""
    out = pd.DataFrame(index=df.index)
    cats = []
    if labels:
        out["catalyst"] = df["catalyst"].astype(str)
        cats.append("catalyst")
    if desc:
        d = pd.DataFrame([descriptors(c) for c in df["catalyst"]], index=df.index)
        out["metal"] = d["metal"]
        out["support"] = d["support"]
        cats += ["metal", "support"]
        for k in ("loading_wt", "ox_state", "bet_m2g"):
            out[k] = d[k].astype(float)
    for k in CONT_RAW:
        out[k] = df[k].astype(float)
    if transformed:
        out["inv_T"] = 1000.0 / (df["T_C"].astype(float) + 273.15)
        out["ln_t"] = np.log(df["t_h"].astype(float))
        out["ln_P"] = np.log(df["P_bar"].astype(float))
        out["ln_cat"] = np.log(df["cat_mol"].astype(float))
    p = df["promoter_wt"].astype(float)
    if promoter == "flag":
        used = df["promoter_used"].astype(bool)
        out["promoter_wt"] = np.where(~used, 0.0, p)  # not-recorded blanks stay NaN
        out["promoter_used"] = used.astype(float)
    elif promoter == "nan":
        out["promoter_wt"] = p
    elif promoter == "mean":
        out["promoter_wt"] = p.fillna(p.mean())
    else:
        raise ValueError(promoter)
    out["solvent"] = df["solvent"].astype(str)
    cats.append("solvent")
    if lab and "lab" in df:
        out["lab"] = df["lab"].astype(str)
        cats.append("lab")
    if leak:
        out["residual_precursor_pct"] = df["residual_precursor_pct"].astype(float)
    return out, cats
