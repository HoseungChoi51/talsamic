"""Harder benchmark problems: real high-throughput reaction-yield datasets.

They are harder than the fake plant in the ways that matter for this project:
* the conditions are chemicals (ligand, base, additive, solvent). A tabular model only
  sees category codes, while an LLM knows what the chemicals are, which is where the
  LLM layer can add something;
* the "held-out chemical" splits ask for yields with a ligand/additive the model has
  never seen, the situation of a plant trying a new catalyst;
* the ``n100`` splits give the model 100 runs, the size of a plant's history.

Datasets (downloaded once from GitHub and cached in ~/.cache/yieldengine):
* buchwald_hartwig: Ahneman et al., Science 2018. 3955 C-N couplings:
  4 ligands x 3 bases x 15 aryl halides x 22 additives. Splits from the paper: additive
  out-of-sample tests 1-4.
* suzuki: Perera et al., Science 2018. 5760 Suzuki-Miyaura couplings: 7 electrophiles,
  4 nucleophiles, 12 ligand options, 8 base options, 6 solvent options.
* direct_arylation: Shields et al., Nature 2021 (EDBO). 1728 C-H arylations:
  12 ligands, 4 bases, 4 solvents, 3 concentrations, 3 temperatures (has numbers
  as well as chemicals, like a plant recipe).
"""

from __future__ import annotations

import os
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Kind, Role, Schema, Variable

RAW = "https://raw.githubusercontent.com"
SOURCES = {
    "buchwald_hartwig": f"{RAW}/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx",
    "suzuki": f"{RAW}/rxn4chemistry/rxn_yields/master/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx",
    "direct_arylation": f"{RAW}/b-shields/edbo/master/experiments/data/direct_arylation/experiment_index.csv",
}


@dataclass
class Benchmark:
    name: str
    data: pd.DataFrame
    schema: Schema
    splits: dict[str, tuple[np.ndarray, np.ndarray]]  # split name -> (train rows, test rows)
    smiles: dict[str, dict[str, str]] = field(default_factory=dict)  # variable -> {category -> SMILES}
    description: str = ""
    name_suffix: str = ""  # "+llm" when LLM descriptors are attached

    def split(self, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        tr, te = self.splits[name]
        return self.data.iloc[tr].reset_index(drop=True), self.data.iloc[te].reset_index(drop=True)


def _cache_dir() -> Path:
    d = Path(os.environ.get("YIELDENGINE_DATA", Path.home() / ".cache" / "yieldengine"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(name: str) -> Path:
    url = SOURCES[name]
    path = _cache_dir() / url.rsplit("/", 1)[1]
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _schema(df: pd.DataFrame, categorical: dict[str, str], continuous: dict[str, tuple[str, str]]) -> Schema:
    vs = []
    for col, label in categorical.items():
        cats = sorted(df[col].astype(str).unique())
        vs.append(Variable(col, label, Kind.CATEGORICAL, Role.SET_BEFORE, categories=cats,
                           standard=df[col].astype(str).mode().iloc[0]))
    for col, (label, unit) in continuous.items():
        x = df[col].astype(float)
        vals = np.sort(x.unique())
        step = float(np.min(np.diff(vals))) if len(vals) > 1 else None
        vs.append(Variable(col, label, Kind.CONTINUOUS, Role.SET_BEFORE, float(x.min()), float(x.max()),
                           standard=float(np.median(vals)), unit=unit, step=step))
    vs.append(Variable("yield", "수율", Kind.CONTINUOUS, Role.TARGET, 0, 100, unit="%", step=0.1))
    return Schema(vs, date_column=None, reference_column=None)


def _random(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.random.default_rng(seed).permutation(n)
    k = int(round(n * frac))
    return np.sort(p[:k]), np.sort(p[k:])


def _small(n: int, size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    p = np.random.default_rng(seed).permutation(n)
    return np.sort(p[:size]), np.sort(p[size:])


def _group_out(df: pd.DataFrame, col: str, k: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """k folds over the categories of ``col``: train never sees the held-out categories."""
    cats = sorted(df[col].astype(str).unique())
    out = {}
    for i in range(k):
        held = set(cats[i::k])
        mask = df[col].astype(str).isin(held).to_numpy()
        out[f"{col}_out{i + 1}"] = (np.where(~mask)[0], np.where(mask)[0])
    return out


def _standard_splits(df: pd.DataFrame) -> dict:
    n = len(df)
    return {"random70": _random(n, 0.7, 0), "n100": _small(n, 100, 0), "n100b": _small(n, 100, 1)}


# -- Buchwald-Hartwig ---------------------------------------------------------------------
_BH_LIGANDS = {  # rules on the SMILES -> common name
    "AdBrettPhos": lambda s: "[C@@]3(C[C@@H]4C5)" in s,
    "tBuBrettPhos": lambda s: "P(C(C)(C)C)C(C)(C)C" in s and "OC" in s,
    "tBuXPhos": lambda s: "P(C(C)(C)C)C(C)(C)C" in s,
    "XPhos": lambda s: "P(C3CCCCC3)C4CCCCC4" in s,
}
_BH_BASES = {"CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC": "P2Et", "CN1CCCN2C1=NCCC2": "MTBD",
             "CC(C)(C)/N=C(N(C)C)/N(C)C": "BTMG"}
_HALO = {"Cl": "chloro", "Br": "bromo", "I": "iodo"}


def _bh_halide(s: str) -> str:
    x = _HALO[s[:2] if s.startswith(("Cl", "Br")) else s[0]]
    if "C(F)(F)F" in s:
        return f"4-{x}benzotrifluoride"
    if "CC)C=C1" in s:
        return f"1-ethyl-4-{x}benzene"
    if "OC)C=C1" in s:
        return f"4-{x}anisole"
    if "=CN=CC=C1" in s:
        return f"3-{x}pyridine"
    if "=NC=CC=C1" in s:
        return f"2-{x}pyridine"
    return s


def buchwald_hartwig() -> Benchmark:
    path = _fetch("buchwald_hartwig")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xl = pd.ExcelFile(path)
        raw = pd.read_excel(xl, "FullCV_01")
        tests = {s: pd.read_excel(xl, s) for s in ("Test1", "Test2", "Test3", "Test4")}
    lig = {s: next(n for n, rule in _BH_LIGANDS.items() if rule(s)) for s in raw["Ligand"].unique()}
    add_ids = {s: f"A{i + 1:02d}" for i, s in enumerate(sorted(raw["Additive"].unique()))}
    df = pd.DataFrame({
        "ligand": raw["Ligand"].map(lig), "base": raw["Base"].map(_BH_BASES),
        "aryl_halide": raw["Aryl halide"].map(_bh_halide), "additive": raw["Additive"].map(add_ids),
        "yield": raw["Output"].astype(float).clip(0, 100),
    })
    schema = _schema(df, {"ligand": "리간드", "base": "염기", "aryl_halide": "할로젠화 아릴", "additive": "첨가제"}, {})
    splits = _standard_splits(df)
    # the paper's additive out-of-sample tests: the first N rows of each sheet are training
    for (sheet, t), n_train in zip(tests.items(), (3057, 3055, 3058, 3055)):
        train_adds = set(t["Additive"].iloc[:n_train])
        mask = raw["Additive"].isin(train_adds).to_numpy()
        splits[f"additive_{sheet.lower()}"] = (np.where(mask)[0], np.where(~mask)[0])
    smiles = {"ligand": {v: k for k, v in lig.items()}, "base": {v: k for k, v in _BH_BASES.items()},
              "aryl_halide": {_bh_halide(s): s for s in raw["Aryl halide"].unique()},
              "additive": {v: k for k, v in add_ids.items()}}
    return Benchmark("buchwald_hartwig", df, schema, splits, smiles, "Ahneman 2018 C-N coupling, 3955 reactions")


# -- Suzuki-Miyaura ---------------------------------------------------------------------
# The file names ligands only by shorthand. Structures help the LLM: without them it took
# AmPhos for a biaryl phosphine and CataCXium A (PAd2nBu) for a ferrocene bisphosphine.
_AD = "C12CC3CC(CC(C3)C1)C2"  # 1-adamantyl
_SZ_LIGANDS = {
    "AmPhos": "CN(C)c1ccc(cc1)P(C(C)(C)C)C(C)(C)C",
    "CataCXium A": f"CCCCP({_AD}){_AD}",
    "P(Cy)3": "C1CCC(CC1)P(C1CCCCC1)C1CCCCC1",
    "P(Ph)3": "c1ccc(cc1)P(c1ccccc1)c1ccccc1",
    "P(o-Tol)3": "Cc1ccccc1P(c1ccccc1C)c1ccccc1C",
    "P(tBu)3": "CC(C)(C)P(C(C)(C)C)C(C)(C)C",
    "SPhos": "COc1cccc(OC)c1-c1ccccc1P(C1CCCCC1)C1CCCCC1",
    "XPhos": "CC(C)c1cc(C(C)C)c(-c2ccccc2P(C2CCCCC2)C2CCCCC2)c(C(C)C)c1",
    "Xantphos": "CC1(C)c2cccc(P(c3ccccc3)c3ccccc3)c2Oc2c(P(c3ccccc3)c3ccccc3)cccc21",
    "dppf": "[Fe+2].c1ccc(cc1)P(c1ccccc1)[c-]1cccc1.c1ccc(cc1)P(c1ccccc1)[c-]1cccc1",
    "dtbpf": "[Fe+2].CC(C)(C)P(C(C)(C)C)[c-]1cccc1.CC(C)(C)P(C(C)(C)C)[c-]1cccc1",
}


def suzuki() -> Benchmark:
    path = _fetch("suzuki")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = pd.read_excel(path, "Sheet1")
    df = pd.DataFrame({
        # names are trimmed: the file has "P(Ph)3 " with a trailing space
        "electrophile": raw["Reactant_1_Name"].astype(str).str.strip(),
        "nucleophile": raw["Reactant_2_Name"].astype(str).str.strip(),
        "ligand": raw["Ligand_Short_Hand"].fillna("None").astype(str).str.strip(),
        "base": raw["Reagent_1_Short_Hand"].fillna("None").astype(str).str.strip(),
        "solvent": raw["Solvent_1_Short_Hand"].fillna("None").astype(str).str.strip(),
        "yield": raw["Product_Yield_PCT_Area_UV"].astype(float).clip(0, 100),
    })
    schema = _schema(df, {"electrophile": "친전자체", "nucleophile": "친핵체", "ligand": "리간드",
                          "base": "염기", "solvent": "용매"}, {})
    splits = {**_standard_splits(df), **_group_out(df, "ligand", 4)}
    return Benchmark("suzuki", df, schema, splits, {"ligand": _SZ_LIGANDS}, "Perera 2018 Suzuki-Miyaura, 5760 reactions")


# -- Direct arylation (EDBO) ------------------------------------------------------------
_DA_LIGANDS = {
    "CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)=CC=C2OC": "BrettPhos",
    "CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C": "PtBu2Ph",
    "CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3": "PhPtBu-CPhos",
    "P(C1CCCCC1)(C2CCCCC2)C3CCCCC3": "PCy3",
    "P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3": "PPh3",
    "CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C": "XPhos",
    "P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3": "P(2-furyl)3",
    "CP(C1=CC=CC=C1)C2=CC=CC=C2": "PPh2Me",
    "CC(OC1=C(P(C2CCCCC2)C3CCCCC3)C(OC(C)C)=CC=C1)C": "RuPhos",
    "FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)=CC=C2OC)C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1": "JackiePhos",
    "C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1": "CgMe-PPh",
    "CP(C)C1=CC=CC=C1": "PPhMe2",
}
_DA_BASES = {"O=C([O-])C.[K+]": "KOAc", "O=C([O-])C(C)(C)C.[K+]": "KOPiv",
             "O=C([O-])C.[Cs+]": "CsOAc", "O=C([O-])C(C)(C)C.[Cs+]": "CsOPiv"}
_DA_SOLVENTS = {"CC(N(C)C)=O": "DMAc", "CCCC#N": "butyronitrile", "CCCCOC(C)=O": "butyl acetate",
                "CC1=CC=C(C)C=C1": "p-xylene"}


def direct_arylation() -> Benchmark:
    raw = pd.read_csv(_fetch("direct_arylation"))
    df = pd.DataFrame({
        "ligand": raw["Ligand_SMILES"].map(_DA_LIGANDS), "base": raw["Base_SMILES"].map(_DA_BASES),
        "solvent": raw["Solvent_SMILES"].map(_DA_SOLVENTS),
        "concentration": raw["Concentration"].astype(float), "temp": raw["Temp_C"].astype(float),
        "yield": raw["yield"].astype(float).clip(0, 100),
    })
    if df[["ligand", "base", "solvent"]].isna().any().any():
        raise ValueError("unmapped SMILES in direct arylation data")
    schema = _schema(df, {"ligand": "리간드", "base": "염기", "solvent": "용매"},
                     {"concentration": ("농도", " M"), "temp": ("반응 온도", "℃")})
    splits = {**_standard_splits(df), **_group_out(df, "ligand", 4)}
    smiles = {"ligand": {v: k for k, v in _DA_LIGANDS.items()}, "base": {v: k for k, v in _DA_BASES.items()},
              "solvent": {v: k for k, v in _DA_SOLVENTS.items()}}
    return Benchmark("direct_arylation", df, schema, splits, smiles, "Shields 2021 (EDBO) C-H arylation, 1728 reactions")


LOADERS = {"buchwald_hartwig": buchwald_hartwig, "suzuki": suzuki, "direct_arylation": direct_arylation}


def load(name: str, descriptors: str | Path | None = None) -> Benchmark:
    """Load a benchmark; with ``descriptors`` (a folder of LLM descriptor tables named
    <benchmark>.json), its categorical conditions also carry those descriptors."""
    bench = LOADERS[name]()
    if descriptors:
        from ..llm.descriptors import attach, load_tables

        path = Path(descriptors) / f"{name}.json"
        bench.schema = attach(bench.schema, load_tables(path))
        bench.name_suffix = "+llm"
    return bench
