"""다음 실험: the batch suggestion pipeline.

1. Sample thousands of candidate recipes inside the constraints: start from the standard
   recipe (with fixed conditions applied) and change at most ``max_changes`` conditions.
2. Score them all in one model call (past runs as context, candidates as queries).
3. Rank by the risk setting and drop near-duplicates; candidates further than the risk
   setting allows from any past run are dropped first.
4. Add "learn" runs: runs that separate conditions that always moved together, then the
   most uncertain nearby candidates. Add the weekly reference run.
5. Attach a two-sentence reason built only from numbers computed here. An LLM may
   rephrase it later, but ``explain.checked_reason`` rejects any number not in ``facts``.

The model never picks the batch size, risk, or constraints, and an LLM never picks runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import norm

from .analysis.importance import find_groups
from .analysis.support import control_distance, nearest_distance
from .korean import josa
from .models import Prediction, Surrogate
from .schema import Schema, Variable

OPTIMIZE, LEARN, REFERENCE = "최적화", "학습", "기준"


@dataclass(frozen=True)
class RiskLevel:
    label: str
    rank_by: str  # "quantile" or "ei"
    quantile: float | None
    radius: float  # max distance from the nearest past run (1 = one condition across its whole range)
    max_new: int | None  # optimize runs per batch that may use a value never run before (None = no limit)


RISK_LEVELS = {
    "안전하게": RiskLevel("안전하게", "quantile", 0.25, 0.15, 0),
    "균형": RiskLevel("균형", "ei", None, 0.30, 1),
    "도전적으로": RiskLevel("도전적으로", "quantile", 0.90, 0.60, None),
}


@dataclass
class BatchSettings:
    risk: str = "균형"  # 모험 정도
    n_runs: int = 12  # 실험 수
    split: tuple[int, int, int] | None = None  # (optimize, learn, reference); default ~ 8/3/1
    fixed: dict[str, Any] = field(default_factory=dict)  # 고정할 조건
    limits: dict[str, Any] = field(default_factory=dict)  # 범위 제한: (lo, hi) or allowed categories
    max_changes: int = 2  # 한 번에 바꿀 조건 수 (fixed conditions don't count)
    n_candidates: int = 4000
    min_separation: float = 0.08
    radius: float | None = None  # override the risk level's distance limit (benchmarks)
    max_new: int | None = None  # override the risk level's limit on optimize runs with a never-run value
    seed: int = 0

    def resolved_split(self) -> tuple[int, int, int]:
        if self.split is not None:
            if sum(self.split) != self.n_runs:
                raise ValueError(f"split {self.split} does not add up to {self.n_runs} runs")
            return tuple(self.split)
        ref = 1 if self.n_runs >= 3 else 0
        learn = int(round(0.25 * (self.n_runs - ref)))
        return self.n_runs - ref - learn, learn, ref


@dataclass
class Suggestion:
    run: int
    purpose: str
    recipe: dict[str, Any]
    changes: list[tuple[str, Any, Any]]  # (name, standard value, new value)
    predicted: float
    low: float  # 80% range, calibrated
    high: float
    reason: str
    facts: dict[str, Any]


@dataclass
class Batch:
    batch_id: str
    created_at: str
    model: str
    settings: BatchSettings
    context: dict[str, Any]
    suggestions: list[Suggestion]
    notes: list[str] = field(default_factory=list)  # e.g. fewer runs than asked, and why

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for s in self.suggestions:
            rows.append({"batch_id": self.batch_id, "run": s.run, "purpose": s.purpose, **s.recipe,
                         **self.context, "predicted": round(s.predicted, 1), "low": round(s.low, 1),
                         "high": round(s.high, 1), "reason": s.reason})
        return pd.DataFrame(rows)

    def to_dict(self) -> dict:
        return {"batch_id": self.batch_id, "created_at": self.created_at, "model": self.model,
                "settings": asdict(self.settings), "context": self.context,
                "suggestions": [asdict(s) for s in self.suggestions], "notes": self.notes}


# --------------------------------------------------------------------------------------
def _bounds(v: Variable, limits: dict) -> tuple[float, float]:
    lo, hi = v.lower, v.upper
    if v.name in limits:
        lo, hi = max(lo, limits[v.name][0]), min(hi, limits[v.name][1])
    return lo, hi


def _allowed(v: Variable, limits: dict) -> list[str]:
    return [c for c in v.categories if c in limits.get(v.name, v.categories)]


def _snap(v: Variable, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if v.step:
        x = np.round(x / v.step) * v.step
    return np.clip(x, lo, hi)


def base_recipe(schema: Schema, settings: BatchSettings) -> dict[str, Any]:
    base = schema.standard_recipe()
    base.update(settings.fixed)
    for v in schema.controls:  # pull the standard recipe inside hard limits if needed
        if v.is_continuous:
            lo, hi = _bounds(v, settings.limits)
            base[v.name] = float(np.clip(base[v.name], lo, hi))
        elif base[v.name] not in _allowed(v, settings.limits):
            base[v.name] = _allowed(v, settings.limits)[0]
    return base


def sample_candidates(schema: Schema, settings: BatchSettings, rng: np.random.Generator) -> pd.DataFrame:
    base = base_recipe(schema, settings)
    free = [v for v in schema.controls if v.name not in settings.fixed]
    free = [v for v in free if v.is_continuous or len(_allowed(v, settings.limits)) > 1]
    n = settings.n_candidates
    cols = {k: np.repeat(np.array([val], dtype=object), n) for k, val in base.items()}
    kmax = max(1, min(settings.max_changes, len(free)))
    n_change = rng.integers(1, kmax + 1, n)
    for i in range(n):
        for j in rng.choice(len(free), n_change[i], replace=False):
            v = free[j]
            if v.is_continuous:
                lo, hi = _bounds(v, settings.limits)
                cols[v.name][i] = float(_snap(v, rng.uniform(lo, hi, 1), lo, hi)[0])
            else:
                opts = [c for c in _allowed(v, settings.limits) if c != base[v.name]]
                cols[v.name][i] = rng.choice(opts)
    df = pd.DataFrame(cols)
    for v in schema.controls:
        df[v.name] = df[v.name].astype(float) if v.is_continuous else df[v.name].astype(str)
    return df.drop_duplicates(ignore_index=True)


def filter_pool(schema: Schema, settings: BatchSettings, pool: pd.DataFrame) -> pd.DataFrame:
    """Keep pool recipes that respect fixed conditions, limits and the change limit."""
    base = base_recipe(schema, settings)
    df = pool[[v.name for v in schema.controls]].drop_duplicates().reset_index(drop=True)
    keep = np.ones(len(df), bool)
    n_changed = np.zeros(len(df), int)
    for v in schema.controls:
        col = df[v.name]
        if v.is_continuous:
            lo, hi = _bounds(v, settings.limits)
            x = col.astype(float).to_numpy()
            keep &= (x >= lo - 1e-9) & (x <= hi + 1e-9)
            same = np.abs(x - float(base[v.name])) < 1e-9
        else:
            keep &= col.astype(str).isin(_allowed(v, settings.limits)).to_numpy()
            same = (col.astype(str) == str(base[v.name])).to_numpy()
        if v.name in settings.fixed:
            keep &= same
        else:
            n_changed += ~same
    keep &= n_changed <= settings.max_changes
    return df[keep].reset_index(drop=True)


def score(pred: Prediction, risk: RiskLevel, incumbent: float) -> np.ndarray:
    if risk.rank_by == "ei":
        z = (pred.mean - incumbent) / pred.sd
        return (pred.mean - incumbent) * norm.cdf(z) + pred.sd * norm.pdf(z)
    return pred.quantile(risk.quantile)


def _pick_diverse(schema, pool: pd.DataFrame, order: np.ndarray, k: int, taken: list[pd.DataFrame],
                  min_sep: float, capped: np.ndarray | None = None, cap: int | None = None) -> list[int]:
    """Greedy by score, skipping near-duplicates, and taking at most ``cap`` rows marked in
    ``capped``. If the constraints leave too little room, the minimum separation is relaxed
    step by step, and only then the cap (exact duplicates are never allowed)."""
    ref = pd.concat(taken, ignore_index=True) if taken else None
    chosen: list[int] = []
    for limit in ([cap, None] if cap is not None and capped is not None else [None]):
        for sep in (min_sep, min_sep / 2, min_sep / 4, 1e-9):
            for i in order:
                if len(chosen) >= k:
                    return chosen
                if i in chosen:
                    continue
                if limit is not None and capped[i] and int(capped[chosen].sum()) >= limit:
                    continue
                others = ([ref] if ref is not None else []) + ([pool.iloc[chosen]] if chosen else [])
                if others and control_distance(schema, pool.iloc[[i]], pd.concat(others, ignore_index=True)).min() < sep:
                    continue
                chosen.append(int(i))
    return chosen


def _uses_new_value(pred: Prediction, settings: BatchSettings) -> np.ndarray:
    """Rows with a never-run value the engineer didn't ask for (fixed conditions don't count)."""
    if pred.novel is None:
        return np.zeros(len(pred), bool)
    return np.array([any(n not in settings.fixed for n in names) for names in pred.novel], bool)


def _py(x):
    return x.item() if isinstance(x, np.generic) else x


def _changes(schema: Schema, recipe: dict, standard: dict) -> list[tuple[str, Any, Any]]:
    out = []
    for v in schema.controls:
        a, b = standard[v.name], recipe[v.name]
        same = abs(float(a) - float(b)) < 1e-9 if v.is_continuous else str(a) == str(b)
        if not same:
            out.append((v.name, a, b))
    return out


def _describe_changes(schema: Schema, changes) -> str:
    if not changes:
        return "표준 조건 그대로"
    return ", ".join(f"{schema[n].label} {schema[n].fmt(a)}→{schema[n].fmt(b)}" for n, a, b in changes)


# --------------------------------------------------------------------------------------
def suggest_batch(
    model: Surrogate,
    data: pd.DataFrame,
    schema: Schema,
    settings: BatchSettings | None = None,
    context: dict[str, Any] | None = None,
    candidates: pd.DataFrame | None = None,
) -> Batch:
    """``model`` should be fitted (ideally a ``CalibratedModel``); ``context`` holds the
    current lot/date values to predict for, e.g. ``FakePlant.context_now(history)``.

    ``candidates``: a fixed pool of recipes to choose from instead of sampling (e.g. the
    combinations a plant can actually run, or a benchmark's untested reactions). The same
    constraints apply; runs that separate aliased conditions are skipped because they
    may not be in the pool."""
    settings = settings or BatchSettings()
    risk = RISK_LEVELS[settings.risk]
    rng = np.random.default_rng(settings.seed)
    context = dict(context or {})
    n_opt, n_learn, n_ref = settings.resolved_split()
    standard = schema.standard_recipe()
    base = base_recipe(schema, settings)
    past = data[[v.name for v in schema.controls]]

    if candidates is None:
        cands = sample_candidates(schema, settings, rng)
    else:
        cands = filter_pool(schema, settings, candidates)
        if cands.empty:
            raise ValueError("no candidate in the pool satisfies the fixed conditions, limits and max changes")
    cands = cands.assign(**context)
    pred = model.predict(cands)  # one call scores every candidate
    dist = nearest_distance(schema, cands, past)

    # incumbent for EI: best predicted yield among recipes already run, under today's context
    past_now = past.drop_duplicates().assign(**context)
    incumbent = float(model.predict(past_now).mean.max())
    std_row = pd.DataFrame([standard]).assign(**context)
    std_pred = float(model.predict(std_row).mean[0])

    radius = settings.radius if settings.radius is not None else risk.radius
    inside = dist <= radius
    if inside.sum() < 5 * max(1, n_opt):  # too few inside the radius: keep the closest
        inside = dist <= np.quantile(dist, min(1.0, 5 * max(1, n_opt) / len(dist)))
    s = score(pred, risk, incumbent)
    s = np.where(inside, s, -np.inf)
    # A never-run value gets a wide range from the new-chemical guard, which pulls 균형 and
    # 도전적으로 toward it. The risk level limits how many optimize runs may be such guesses;
    # trying new values is otherwise the learn runs' job.
    new_value = _uses_new_value(pred, settings)
    max_new = settings.max_new if settings.max_new is not None else risk.max_new
    opt_idx = _pick_diverse(schema, cands, np.argsort(-s)[: int(inside.sum())], n_opt, [], settings.min_separation,
                            new_value, max_new)

    suggestions: list[Suggestion] = []
    lo80, hi80 = pred.interval(0.8)

    def add(purpose, recipe, mean, low, high, facts, reason):
        changes = _changes(schema, recipe, standard)
        suggestions.append(Suggestion(len(suggestions) + 1, purpose, recipe, changes,
                                      float(mean), float(low), float(high), reason, facts))

    for i in opt_idx:
        recipe = {v.name: _py(cands.iloc[i][v.name]) for v in schema.controls}
        facts = {"predicted": round(float(pred.mean[i]), 1), "low": round(float(lo80[i]), 1),
                 "high": round(float(hi80[i]), 1), "vs_standard_pp": round(float(pred.mean[i]) - std_pred, 1),
                 "nearest_past_distance": round(float(dist[i]), 2)}
        desc = _describe_changes(schema, _changes(schema, recipe, standard))
        reason = (f"{desc}: 예상 수율 {facts['predicted']}% (범위 {facts['low']}–{facts['high']}%), "
                  f"표준보다 {facts['vs_standard_pp']:+.1f}%p. "
                  f"모험 정도 '{risk.label}' 설정으로 고른 조건이다.") + _novel_note(schema, pred, i, facts)
        add(OPTIMIZE, recipe, pred.mean[i], lo80[i], hi80[i], facts, reason)

    # ---- learn runs ---------------------------------------------------------------
    learn_rows = [] if candidates is not None else _disentangle_runs(schema, data, settings, base)[:n_learn]
    if learn_rows:
        lr = pd.DataFrame([r for r, _ in learn_rows]).assign(**context)
        lp = model.predict(lr)
        llo, lhi = lp.interval(0.8)
        for k, (recipe, (a, b)) in enumerate(learn_rows):
            facts = {"predicted": round(float(lp.mean[k]), 1), "low": round(float(llo[k]), 1),
                     "high": round(float(lhi[k]), 1), "separates": [a, b]}
            reason = (f"{josa(schema[a].label, '과')} {josa(schema[b].label, '은')} 늘 같이 움직여 효과를 구분할 수 없었다. "
                      f"{schema[a].label}만 바꿔서 어느 쪽 효과인지 가린다.")
            add(LEARN, recipe, lp.mean[k], llo[k], lhi[k], facts, reason)
    n_left = n_learn - len(learn_rows)
    if n_left > 0:
        wide = dist <= max(min(1.0, radius * 1.5), np.min(dist))
        order = np.argsort(-np.where(wide, pred.sd, -np.inf))[: int(wide.sum())]
        taken = [cands.iloc[opt_idx]] + ([pd.DataFrame([r for r, _ in learn_rows])] if learn_rows else [])
        for i in _pick_diverse(schema, cands, order, n_left, taken, settings.min_separation * 2):
            recipe = {v.name: _py(cands.iloc[i][v.name]) for v in schema.controls}
            half = round(float(hi80[i] - lo80[i]) / 2, 1)
            facts = {"predicted": round(float(pred.mean[i]), 1), "low": round(float(lo80[i]), 1),
                     "high": round(float(hi80[i]), 1), "half_width": half}
            reason = (f"이 조건 근처는 데이터가 적어 예측 범위가 넓다 (±{half}%p). "
                      f"결과가 모델을 가장 많이 바로잡을 수 있는 실험이다.") + _novel_note(schema, pred, i, facts)
            add(LEARN, recipe, pred.mean[i], lo80[i], hi80[i], facts, reason)

    # ---- reference run --------------------------------------------------------------
    for _ in range(n_ref):
        recipe = dict(standard)
        moved = {k: v for k, v in settings.fixed.items() if str(standard.get(k)) != str(v)}
        recipe.update(moved)
        rp = model.predict(pd.DataFrame([recipe]).assign(**context))
        rlo, rhi = rp.interval(0.8)
        facts = {"predicted": round(float(rp.mean[0]), 1), "low": round(float(rlo[0]), 1),
                 "high": round(float(rhi[0]), 1)}
        reason = "매주 같은 표준 조건으로 돌려서 설비나 원료가 변했는지 확인한다." + _novel_note(schema, rp, 0, facts)
        if moved:
            reason += " 고정 조건 때문에 표준과 다른 부분이 있어 지난 기준 실험과 직접 비교하기 어렵다."
        add(REFERENCE, recipe, rp.mean[0], rlo[0], rhi[0], facts, reason)

    notes = []
    n_new = int(new_value[opt_idx].sum()) if len(opt_idx) else 0
    if max_new is not None and n_new > max_new:
        notes.append(f"처음 써 보는 값이 들어간 최적화 실험은 {max_new}건까지로 정했지만, "
                     f"고정 조건과 범위 제한 안에 다른 조건이 부족해 {n_new}건이 됐다.")
    if len(suggestions) < settings.n_runs:
        notes.append(f"고정 조건과 범위 제한 안에서 서로 다른 조건을 {len(suggestions)}건만 찾았다 "
                     f"(요청 {settings.n_runs}건). 제한을 풀거나 한 번에 바꿀 조건 수를 늘리면 더 나온다.")
    # the letter prefix keeps spreadsheets and read_csv from turning an all-digit id into a number
    return Batch("b" + uuid4().hex[:8], datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 getattr(model, "name", type(model).__name__), settings, context, suggestions, notes)


def _novel_note(schema: Schema, pred: Prediction, i: int, facts: dict) -> str:
    """Say so when a recipe uses a value the past runs never used (see ``novelty.py``)."""
    names = pred.novel[i] if pred.novel is not None else ()
    if not names:
        return ""
    facts["new_conditions"] = list(names)
    labels = ", ".join(schema[n].label for n in names)
    return (f" 처음 써 보는 조건({labels})이 있어 예상 수율은 추정이다. "
            f"범위는 지난 데이터에서 새 값을 예측했을 때 틀린 만큼 넓혔다.")


def _disentangle_runs(schema: Schema, data: pd.DataFrame, settings: BatchSettings, base: dict):
    """One run per member of each aliased group: move that member to the far side of its
    observed range and keep the others at the standard value (one change each)."""
    out = []
    groups = [g for g in find_groups(data, schema) if len(g.members) > 1]
    for g in groups:
        members = [m for m in g.members if m in base and m not in settings.fixed and schema[m].is_continuous]
        for m in members:
            v = schema[m]
            lo, hi = _bounds(v, settings.limits)
            q10, q90 = np.quantile(data[m], [0.1, 0.9])
            target = q90 if abs(q90 - base[m]) >= abs(q10 - base[m]) else q10
            recipe = dict(base)
            recipe[m] = float(_snap(v, np.array([target]), lo, hi)[0])
            partner = next(x for x in g.members if x != m)
            out.append((recipe, (m, partner)))
    return out
