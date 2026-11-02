"""LLM-proposed descriptors for categorical conditions (ligand, base, solvent, catalyst...).

A tabular model sees a chemical only as a category code, so it can't say anything about
a ligand it has never seen. An LLM knows chemistry. Here it fills in a short, fixed list
of numeric descriptors per category (the engine chooses the list and the allowed range),
and the model then gets those numbers as extra inputs.

Safeguards, matching the rest of the design:
* the engine chooses which descriptors exist and their ranges; the LLM only fills values;
* values outside the range, or missing, become NaN, and the reason is recorded;
* the LLM is asked ``n_samples`` times; the median is used and the spread is kept, so a
  descriptor the LLM is unsure about is visible;
* the table is saved as JSON, shown to an engineer, and marked ``confirmed_by`` before
  the tool uses it for real suggestions (benchmarks may run unconfirmed tables).

    python -m yieldengine.llm.descriptors --benchmark direct_arylation --out descriptors/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..schema import Schema, Variable
from .client import ChatClient, parse_json


@dataclass(frozen=True)
class Descriptor:
    name: str
    lower: float
    upper: float
    meaning: str


SPECS: dict[str, list[Descriptor]] = {
    "ligand": [
        Descriptor("steric_bulk", 1, 5, "bulk around the metal: 1 small (PMe3) to 5 very bulky (P(tBu)3, AdBrettPhos)"),
        Descriptor("electron_richness", 1, 5, "sigma-donor strength: 1 poor (P(2-furyl)3) to 5 very rich (P(tBu)3, PCy3)"),
        Descriptor("cone_angle_deg", 100, 220, "Tolman cone angle estimate in degrees"),
        Descriptor("biaryl_phosphine", 0, 1, "1 if a Buchwald-type dialkylbiaryl phosphine, else 0"),
        Descriptor("bidentate", 0, 1, "1 if it binds through two donor atoms, else 0"),
    ],
    "base": [
        Descriptor("pka_conjugate_acid", 0, 40, "pKa of the conjugate acid in DMSO (estimate)"),
        Descriptor("strength", 1, 5, "1 weak (acetate) to 5 very strong (alkoxide, phosphazene)"),
        Descriptor("organic", 0, 1, "1 if an organic (soluble, non-metal) base, else 0"),
        Descriptor("cation_size", 0, 5, "0 none/organic, 1 Li, 2 Na, 3 K, 4 Rb, 5 Cs"),
        Descriptor("steric_bulk", 1, 5, "1 small to 5 very hindered"),
    ],
    "solvent": [
        Descriptor("dielectric_constant", 1, 80, "relative permittivity at 25 C"),
        Descriptor("boiling_point_c", 30, 250, "boiling point in C"),
        Descriptor("protic", 0, 1, "1 if protic (has O-H or N-H), else 0"),
        Descriptor("coordinating", 1, 5, "1 non-coordinating (toluene) to 5 strongly coordinating to Pd (DMSO, MeCN)"),
        Descriptor("water_miscible", 0, 1, "1 if fully miscible with water, else 0"),
    ],
    "additive": [
        Descriptor("catalyst_poisoning", 1, 5, "how strongly it can bind Pd and inhibit the catalyst"),
        Descriptor("electron_withdrawing", 1, 5, "electron-withdrawing character of its substituents"),
        Descriptor("hindrance_near_n", 1, 5, "steric hindrance next to the coordinating nitrogen"),
        Descriptor("n_o_bond_lability", 1, 5, "how easily an N-O bond opens (isoxazole ring opening); 1 if none"),
    ],
    "aryl_halide": [
        Descriptor("halide_reactivity", 1, 5, "ease of oxidative addition: Cl 1, Br 3, I 5"),
        Descriptor("ring_electron_poor", 1, 5, "1 electron-rich ring (anisole) to 5 electron-poor (CF3-arene, pyridine)"),
        Descriptor("heteroaryl", 0, 1, "1 if the ring is a heteroarene, else 0"),
        Descriptor("n_adjacent_to_halide", 0, 1, "1 if a ring nitrogen sits next to the C-X carbon, else 0"),
    ],
}
SPECS["electrophile"] = SPECS["aryl_halide"]
GENERIC = [
    Descriptor("reactivity", 1, 5, "general reactivity in this reaction: 1 low to 5 high"),
    Descriptor("steric_bulk", 1, 5, "1 small to 5 very bulky"),
    Descriptor("polarity", 1, 5, "1 non-polar to 5 very polar"),
]


def spec_for(var: Variable) -> list[Descriptor]:
    return SPECS.get(var.name, GENERIC)


SYSTEM = ("You are an expert synthetic and process chemist. You estimate physical-organic descriptors "
          "for reagents from their names and structures. Answer with one JSON object only.")


def _prompt(var: Variable, smiles: dict[str, str], spec: list[Descriptor], context: str) -> str:
    lines = [f"Reaction: {context}" if context else "",
             f"Condition: {var.name} (shown to engineers as '{var.label}').",
             "Estimate these descriptors for every option. Use the scale exactly as defined; "
             "give your best estimate even when unsure, and use null only if it truly doesn't apply.", ""]
    for d in spec:
        lines.append(f"- {d.name}: number from {d.lower:g} to {d.upper:g}. {d.meaning}")
    lines += ["", "Options:"]
    for c in var.categories:
        lines.append(f"- {c}" + (f" (SMILES {smiles[c]})" if c in smiles else ""))
    lines += ["", 'Reply as {"values": {"<option>": {"<descriptor>": <number>, ...}, ...}, '
              '"notes": "<one sentence on anything uncertain>"}']
    return "\n".join(x for x in lines if x is not None)


@dataclass
class DescriptorTable:
    variable: str
    values: dict[str, dict[str, float]]  # descriptor -> category -> median value
    spread: dict[str, dict[str, float]]  # descriptor -> category -> max-min across samples
    spec: list[dict]
    model: str
    n_samples: int
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    created_at: str = ""
    confirmed_by: str | None = None


def propose(var: Variable, client: ChatClient, smiles: dict[str, str] | None = None, context: str = "",
            n_samples: int = 3) -> DescriptorTable:
    spec = spec_for(var)
    smiles = smiles or {}
    samples, issues, notes = [], [], []
    for k in range(n_samples):
        try:
            reply = parse_json(client.chat(SYSTEM, _prompt(var, smiles, spec, context),
                                           temperature=0.0 if k == 0 else 0.4, seed=k))
        except Exception as e:
            issues.append(f"sample {k}: unusable reply ({type(e).__name__}: {e})")
            continue
        notes.append(str(reply.get("notes", "")))
        got = {str(c).strip(): v for c, v in (reply.get("values") or {}).items() if isinstance(v, dict)}
        sample = {}
        for d in spec:
            col = {}
            for c in var.categories:
                x = got.get(str(c).strip(), {}).get(d.name)
                try:
                    x = float(x)
                except (TypeError, ValueError):
                    x = np.nan
                if not np.isnan(x) and not (d.lower <= x <= d.upper):
                    issues.append(f"sample {k}: {c}.{d.name}={x:g} outside {d.lower:g}-{d.upper:g}, dropped")
                    x = np.nan
                col[c] = x
            sample[d.name] = col
        missing = [c for c in var.categories if str(c).strip() not in got]
        if missing:
            issues.append(f"sample {k}: no values for {', '.join(missing)}")
        samples.append(sample)
    if not samples:
        raise RuntimeError(f"the LLM gave no usable descriptors for {var.name}: {issues}")
    values, spread = {}, {}
    for d in spec:
        arr = np.array([[s[d.name][c] for c in var.categories] for s in samples], float)
        with np.errstate(all="ignore"):
            med = np.nanmedian(arr, axis=0)
            rng = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
        values[d.name] = {c: (None if np.isnan(m) else round(float(m), 3)) for c, m in zip(var.categories, med)}
        # with one usable sample there is no spread to measure: None, not a reassuring 0
        spread[d.name] = {c: (None if np.isnan(r) or len(samples) < 2 else round(float(r), 3))
                          for c, r in zip(var.categories, rng)}
    return DescriptorTable(var.name, values, spread, [asdict(d) for d in spec], client.model, len(samples),
                           issues, notes, datetime.now(timezone.utc).isoformat(timespec="seconds"))


def save(tables: list[DescriptorTable], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(t) for t in tables], ensure_ascii=False, indent=2), encoding="utf-8")


def load_tables(path: str | Path) -> list[DescriptorTable]:
    return [DescriptorTable(**t) for t in json.loads(Path(path).read_text(encoding="utf-8"))]


def attach(schema: Schema, tables: list[DescriptorTable], require_confirmed: bool = False) -> Schema:
    """Copy of the schema whose categorical variables carry the descriptor values."""
    by_var = {t.variable: t for t in tables}
    new = []
    for v in schema.variables:
        t = by_var.get(v.name)
        if t is None or v.is_continuous:
            new.append(v)
            continue
        if require_confirmed and not t.confirmed_by:
            raise ValueError(f"descriptors for {v.name} have not been confirmed by an engineer")
        name = {str(c).strip(): c for c in v.categories}  # match names even if one side has stray spaces
        desc = {d: {name[k.strip()]: x for k, x in col.items() if x is not None and k.strip() in name}
                for d, col in t.values.items()}
        desc = {d: col for d, col in desc.items() if len(set(col.values())) > 1}  # constant = no information
        new.append(Variable(**{**v.__dict__, "descriptors": desc}))
    return Schema(new, schema.date_column, schema.reference_column)


def main(argv=None) -> None:
    from ..benchmarks.datasets import LOADERS, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", nargs="+", default=list(LOADERS), choices=list(LOADERS))
    ap.add_argument("--out", default="descriptors")
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--timeout", type=float, help="seconds per LLM call (default 1800, or $YIELDENGINE_LLM_TIMEOUT)")
    ap.add_argument("--variables", nargs="+",
                    help="redo only these conditions and replace their tables in the existing <out>/<benchmark>.json")
    a = ap.parse_args(argv)
    client = ChatClient(**{k: v for k, v in {"url": a.url, "model": a.model, "timeout": a.timeout}.items() if v})
    for name in a.benchmark:
        bench = load(name)
        path = Path(a.out) / f"{name}.json"
        keep = load_tables(path) if a.variables and path.exists() else []
        tables = {t.variable: t for t in keep}
        for v in bench.schema.controls:
            if v.is_continuous or (a.variables and v.name not in a.variables):
                continue
            t = propose(v, client, bench.smiles.get(v.name), bench.description, a.samples)
            print(f"{name}.{v.name}: {len(v.categories)} options, {t.n_samples} samples, {len(t.issues)} issues; "
                  + "; ".join(t.notes)[:300], flush=True)
            tables[v.name] = t
        order = [v.name for v in bench.schema.controls]
        save(sorted(tables.values(), key=lambda t: order.index(t.variable) if t.variable in order else len(order)),
             path)


if __name__ == "__main__":
    main()
