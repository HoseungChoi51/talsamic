"""Grouped permutation importance.

Conditions that always moved together in the past (|Spearman| >= threshold) are one
group: the data cannot say which of them matters, so they are shuffled and scored
together, and the 발견 screen says they cannot be told apart.

Importance is "model reliance": shuffle a group's columns across runs (jointly, keeping
the members paired) and measure how far predicted yield moves, in %p. It needs no
held-out set, so it stays usable on 100-run datasets.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..models import Surrogate
from ..schema import Schema


@dataclass
class Group:
    members: list[str]
    corr: float = 1.0  # weakest pairwise |Spearman| inside the group

    @property
    def key(self) -> str:
        return "+".join(sorted(self.members))

    def label(self, schema: Schema) -> str:
        return "·".join(schema[m].label for m in self.members)


def find_groups(data: pd.DataFrame, schema: Schema, threshold: float = 0.9) -> list[Group]:
    """Group model inputs whose past values moved together. Categoricals stay alone."""
    cont = [v.name for v in schema.features if v.is_continuous and data[v.name].nunique() > 1]
    parent = {n: n for n in cont}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n

    corr = {}
    for i, a in enumerate(cont):
        for b in cont[i + 1:]:
            r = abs(spearmanr(data[a], data[b]).statistic)
            corr[(a, b)] = r
            if r >= threshold:
                parent[root(b)] = root(a)
    clusters: dict[str, list[str]] = {}
    for n in cont:
        clusters.setdefault(root(n), []).append(n)
    groups = []
    for members in clusters.values():
        pairs = [corr[(a, b)] for i, a in enumerate(cont) for b in cont[i + 1:]
                 if a in members and b in members]
        groups.append(Group(members, min(pairs) if pairs else 1.0))
    groups += [Group([v.name]) for v in schema.features if not v.is_continuous]
    order = {v.name: i for i, v in enumerate(schema.features)}
    return sorted(groups, key=lambda g: min(order[m] for m in g.members))


def grouped_importance(
    model: Surrogate, data: pd.DataFrame, groups: list[Group], n_repeats: int = 5, seed: int = 0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = model.predict(data).mean
    blocks, owners = [], []
    for g in groups:
        for _ in range(n_repeats):
            perm = rng.permutation(len(data))
            shuffled = data.copy()
            for m in g.members:
                shuffled[m] = data[m].to_numpy()[perm]
            blocks.append(shuffled)
            owners.append(g.key)
    pred = model.predict(pd.concat(blocks, ignore_index=True)).mean.reshape(len(blocks), len(data))
    moved = np.abs(pred - base[None, :]).mean(axis=1)
    df = pd.DataFrame({"group": owners, "moved": moved}).groupby("group", sort=False)["moved"].agg(["mean", "std"])
    df = df.rename(columns={"mean": "importance", "std": "spread"}).reset_index()
    df["members"] = [next(g.members for g in groups if g.key == k) for k in df["group"]]
    total = df["importance"].sum()
    df["share"] = df["importance"] / total if total > 0 else 0.0
    return df.sort_values("importance", ascending=False, ignore_index=True)
