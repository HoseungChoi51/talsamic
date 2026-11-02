"""발견: the fixed set of analyses run whenever the data changes.

Each result is one plain Korean sentence, a small chart (plain dicts the UI can draw),
the numbers behind it (``facts``, the only numbers an LLM may repeat), and a strength
label instead of a p-value:

* every finding is recomputed on the earlier half and the later half of the runs,
  each with its own refitted model;
* holds in both halves -> 근거 강함, in one -> 보통, in neither -> 약함 (hidden by default).

Engineers then react with 맞아요 / 이상해요 / 모르겠어요; 이상해요 items go on the
weekly meeting agenda.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from ..calibration import rolling_origin_errors, time_order
from ..korean import fmt_pp, josa
from ..models import ModelFactory, Surrogate
from ..schema import Role, Schema
from .drift import error_drift, reference_drift
from .importance import Group, find_groups, grouped_importance
from .partial_dependence import effect_size, partial_dependence

STRONG, MEDIUM, WEAK = "근거 강함", "보통", "약함"
_RANK = {STRONG: 0, MEDIUM: 1, WEAK: 2}


def strength_label(early: bool, late: bool) -> str:
    return STRONG if early and late else MEDIUM if early or late else WEAK


@dataclass
class Finding:
    key: str  # stable id, so judgments survive a re-run
    kind: str  # aliasing | importance | no_effect | interaction | sweet_spot | edge | drift
    sentence: str
    strength: str
    facts: dict[str, Any] = field(default_factory=dict)
    chart: dict[str, Any] = field(default_factory=dict)

    @property
    def show(self) -> bool:
        return self.strength != WEAK

    def to_dict(self) -> dict:
        return {"key": self.key, "kind": self.kind, "sentence": self.sentence, "strength": self.strength,
                "show": self.show, "facts": self.facts, "chart": self.chart}


class Judgment(str, Enum):
    AGREE = "맞아요"
    ODD = "이상해요"
    UNSURE = "모르겠어요"


def meeting_agenda(findings: list[Finding], judgments: dict[str, Judgment | str]) -> list[Finding]:
    """Findings someone marked 이상해요: where domain knowledge meets the statistics."""
    return [f for f in findings if Judgment(judgments.get(f.key, Judgment.UNSURE)) == Judgment.ODD]


# --------------------------------------------------------------------------------------
@dataclass
class _Slice:
    data: pd.DataFrame
    model: Surrogate
    importance: pd.DataFrame | None = None


def discover(
    data: pd.DataFrame,
    schema: Schema,
    factory: ModelFactory,
    errors: pd.DataFrame | None = None,
    min_share: float = 0.08,
    no_effect_share: float = 0.03,
    min_effect: float = 0.8,
    alias_threshold: float = 0.9,
) -> list[Finding]:
    d = time_order(data, schema).reset_index(drop=True)
    half = len(d) // 2
    slices = {
        "full": _Slice(d, factory().fit(d)),
        "early": _Slice(d.iloc[:half], factory().fit(d.iloc[:half])),
        "late": _Slice(d.iloc[half:], factory().fit(d.iloc[half:])),
    }
    groups = find_groups(d, schema, alias_threshold)
    for s in slices.values():
        s.importance = grouped_importance(s.model, s.data, groups)

    out: list[Finding] = []
    out += _aliasing(schema, groups, slices, alias_threshold)
    out += _importance(schema, slices, min_share, no_effect_share)
    out += _shapes(schema, groups, slices, min_effect)
    if errors is None:
        errors = rolling_origin_errors(factory, d, schema)
    out += _drift(schema, d, errors)
    return sorted(out, key=lambda f: (_RANK[f.strength], _KIND_ORDER.index(f.kind)))


_KIND_ORDER = ["drift", "interaction", "edge", "sweet_spot", "importance", "aliasing", "no_effect"]


# --------------------------------------------------------------------------------------
def _aliasing(schema, groups: list[Group], slices, threshold) -> list[Finding]:
    out = []
    for g in groups:
        if len(g.members) < 2:
            continue
        holds = {}
        for k in ("early", "late"):
            sub = find_groups(slices[k].data, schema, threshold)
            holds[k] = any(set(g.members) <= set(h.members) for h in sub)
        labels = [schema[m].label for m in g.members]
        joined = ", ".join(labels[:-1])
        a = josa(joined, "과")
        b = josa(labels[-1], "은")
        sentence = (f"{a} {b} 지금까지 항상 같이 움직여서 (상관 {g.corr:.2f}) "
                    f"어느 쪽이 수율을 바꾸는지 구분할 수 없다.")
        pts = slices["full"].data[g.members[:2]].round(3).to_dict("records")
        out.append(Finding(f"aliasing:{g.key}", "aliasing", sentence, strength_label(holds["early"], holds["late"]),
                           {"members": g.members, "correlation": round(g.corr, 2)},
                           {"type": "scatter", "x": g.members[0], "y": g.members[1], "points": pts}))
    return out


def _share(imp: pd.DataFrame, key: str) -> float:
    row = imp[imp["group"] == key]
    return float(row["share"].iloc[0]) if len(row) else 0.0


def _importance(schema, slices, min_share, no_effect_share) -> list[Finding]:
    full = slices["full"].importance
    chart = {"type": "bar", "unit": "%p",
             "bars": [{"label": "·".join(schema[m].label for m in r.members), "value": round(r.importance, 2),
                       "group": r.group} for r in full.itertuples()]}
    out = []
    for r in full.itertuples():
        members = list(r.members)
        roles = {schema[m].role for m in members}
        if "days" in members:  # time is covered by the drift finding
            continue
        label = "·".join(schema[m].label for m in members)
        tag = "(구분 불가)" if len(members) > 1 else ""
        if r.share >= min_share:
            early = _share(slices["early"].importance, r.group) >= min_share / 2
            late = _share(slices["late"].importance, r.group) >= min_share / 2
            if roles == {Role.CONTEXT}:
                sentence = f"{josa(label, '이')} 바뀌면 수율이 평균 {r.importance:.1f}%p 달라진다."
            else:
                subj = josa(label, "은")
                subj = label + tag + subj[len(label):]
                sentence = (f"{subj} 수율에 영향이 크다: "
                            f"이 조건만 섞어 보면 예측 수율이 평균 {r.importance:.1f}%p 움직인다.")
            out.append(Finding(f"importance:{r.group}", "importance", sentence, strength_label(early, late),
                               {"members": members, "moved_pp": round(r.importance, 1),
                                "share_pct": round(100 * r.share)}, chart))
        elif r.share < no_effect_share and roles == {Role.SET_BEFORE} and len(members) == 1:
            v = schema[members[0]]
            if not v.is_continuous:
                continue
            x = slices["full"].data[v.name]
            early = _share(slices["early"].importance, r.group) < 2 * no_effect_share
            late = _share(slices["late"].importance, r.group) < 2 * no_effect_share
            lo, hi = v.fmt(x.min()), v.fmt(x.max())
            sentence = f"{josa(v.label, '은')} 지금까지 써 본 범위({lo}–{hi})에서는 수율과 거의 관계없다."
            out.append(Finding(f"no_effect:{v.name}", "no_effect", sentence, strength_label(early, late),
                               {"variable": v.name, "moved_pp": round(r.importance, 2),
                                "observed_min": float(x.min()), "observed_max": float(x.max())}, chart))
    return out


# --------------------------------------------------------------------------------------
def _interaction(schema, model, data, var, cat, min_effect, min_rows=8):
    """Return (dominant category, its effect, best other effect, per-category PD) or None."""
    curves, effects = {}, {}
    for c in schema[cat].categories:
        if (data[cat].astype(str) == c).sum() < min_rows:
            continue
        curves[c] = partial_dependence(model, data, schema, var, at={cat: c})
        effects[c] = effect_size(curves[c])
    if len(effects) < 2:
        return None
    ranked = sorted(effects.items(), key=lambda kv: -kv[1])
    (c1, e1), (_, e2) = ranked[0], ranked[1]
    if e1 >= min_effect and e1 >= 2 * max(e2, 0.1):
        return c1, e1, e2, curves
    return None


def _shape_claim(pdf: pd.DataFrame, span: float, min_effect: float):
    """Classify a PD curve: ('sweet_spot', lo, hi, best) / ('edge', direction, edge, best) / None."""
    s = pdf[pdf["supported"]].reset_index(drop=True)
    if len(s) < 3 or effect_size(pdf) < min_effect:
        return None
    i = int(s["mean"].idxmax())
    best = float(s["mean"].iloc[i])
    if 0 < i < len(s) - 1:
        tol = max(0.3, 0.15 * effect_size(pdf))
        near = s[s["mean"] >= best - tol]["value"]
        return ("sweet_spot", float(near.min()), float(near.max()), best, float(s["value"].iloc[i]))
    return ("edge", "up" if i == len(s) - 1 else "down", float(s["value"].iloc[i]), best, float(s["value"].iloc[i]))


def _same_shape(a, b, span) -> bool:
    if a is None or b is None or a[0] != b[0]:
        return False
    if a[0] == "edge":
        return a[1] == b[1]
    return abs(a[4] - b[4]) <= 0.15 * span


def _curve_chart(pdf: pd.DataFrame, var: str, at: dict | None = None) -> dict:
    return {"type": "line", "x": var, "at": at or {},
            "points": pdf[["value", "mean", "lo", "hi", "supported"]].round(3).to_dict("records")}


def _shapes(schema: Schema, groups, slices, min_effect) -> list[Finding]:
    aliased = {m for g in groups if len(g.members) > 1 for m in g.members}
    cats = [v.name for v in schema.controls if not v.is_continuous]
    out = []
    full = slices["full"]
    for v in schema.controls:
        if not v.is_continuous or v.name in aliased:
            continue
        at: dict = {}
        for cat in cats:
            hit = _interaction(schema, full.model, full.data, v.name, cat, min_effect)
            if hit is None:
                continue
            c1, e1, e2, curves = hit
            holds = {}
            for k in ("early", "late"):
                h = _interaction(schema, slices[k].model, slices[k].data, v.name, cat, min_effect / 2)
                holds[k] = h is not None and h[0] == c1
            others = [c for c in curves if c != c1]
            sentence = (f"{v.label} 효과는 {schema[cat].label} {c1}에서만 크다 "
                        f"({c1}: {e1:.1f}%p, {'·'.join(others)}: {e2:.1f}%p 이하).")
            chart = {"type": "lines", "x": v.name, "series": {c: _curve_chart(p, v.name)["points"]
                                                              for c, p in curves.items()}}
            out.append(Finding(f"interaction:{v.name}|{cat}", "interaction", sentence,
                               strength_label(holds["early"], holds["late"]),
                               {"variable": v.name, "by": cat, "category": c1,
                                "effect_pp": round(e1, 1), "other_effect_pp": round(e2, 1)}, chart))
            at = {cat: c1}  # describe the curve's shape where the effect lives
            break

        claims = {}
        curves = {}
        for k, s in slices.items():
            curves[k] = partial_dependence(s.model, s.data, schema, v.name, at=at)
            claims[k] = _shape_claim(curves[k], v.span(), min_effect if k == "full" else min_effect / 2)
        c = claims["full"]
        if c is None:
            continue
        where = "".join(f"{schema[k].label} {val}에서 " for k, val in at.items())
        early, late = (_same_shape(c, claims[k], v.span()) for k in ("early", "late"))
        key_at = "".join(f"@{k}={val}" for k, val in at.items())
        if c[0] == "sweet_spot":
            lo, hi = v.fmt(c[1]), v.fmt(c[2])
            sentence = f"{where}{josa(v.label, '은')} {lo}–{hi} 구간에서 수율이 가장 높다."
            facts = {"variable": v.name, "at": at, "low": round(c[1], 2), "high": round(c[2], 2),
                     "best_predicted_yield": round(c[3], 1)}
        else:
            word = "높일수록" if c[1] == "up" else "낮출수록"
            sentence = (f"{where}{josa(v.label, '을')} {word} 수율이 오르고, 데이터가 있는 끝"
                        f"({v.fmt(c[2])})까지 계속 오른다. 그 너머는 아직 모른다.")
            facts = {"variable": v.name, "at": at, "direction": c[1], "edge": round(c[2], 2),
                     "best_predicted_yield": round(c[3], 1)}
        out.append(Finding(f"{c[0]}:{v.name}{key_at}", c[0], sentence, strength_label(early, late),
                           facts, _curve_chart(curves["full"], v.name, at)))
    return out


# --------------------------------------------------------------------------------------
def _drift(schema: Schema, d: pd.DataFrame, errors: pd.DataFrame) -> list[Finding]:
    days = d["days"].to_numpy(float) if "days" in d else np.arange(len(d), dtype=float)
    e = error_drift(errors, days)
    r = reference_drift(d, schema)
    flagged = [s for s in (e, r) if s is not None and s.flagged]
    if not flagged:
        return []
    parts, facts = [], {}
    if r is not None and r.flagged:
        word = "낮다" if r.delta < 0 else "높다"
        parts.append(f"기준 실험 수율이 최근 {int(r.detail['window_days'] // 7)}주 평균 "
                     f"{r.detail['recent_mean']:.1f}%로 이전({r.detail['baseline_mean']:.1f}%)보다 {fmt_pp(r.delta)} {word}")
        facts.update(reference_recent=round(r.detail["recent_mean"], 1),
                     reference_before=round(r.detail["baseline_mean"], 1), reference_delta_pp=round(r.delta, 1),
                     reference_window_weeks=int(r.detail["window_days"] // 7))
    if e is not None and e.flagged:
        word = "낮다" if e.delta < 0 else "높다"
        parts.append(f"최근 {int(e.detail['window_days'] // 7)}주 수율이 모델 예상보다 평균 {fmt_pp(e.delta)} {word}")
        facts.update(error_delta_pp=round(e.delta, 1), error_window_weeks=int(e.detail["window_days"] // 7))
    sentence = "설비나 원료가 변했을 수 있다: " + ", ".join(parts) + "."
    strength = STRONG if len(flagged) == 2 else MEDIUM
    chart = {"type": "drift",
             "reference": r.detail["series"] if r is not None else [],
             "weekly_error": e.detail["weekly_error"] if e is not None else {}}
    return [Finding("drift:recent", "drift", sentence, strength, facts, chart)]
