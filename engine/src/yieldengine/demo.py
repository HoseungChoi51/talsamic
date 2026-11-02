"""End-to-end demo on fake data: findings, a batch, run it on the fake plant, compare.

    python -m yieldengine.demo [--model gp|limix-16m] [--risk 균형] [--out demo_out]
"""

from __future__ import annotations

import argparse
import warnings

import pandas as pd

from .engine import Engine
from .simulator import FakePlant, default_schema
from .suggest import BatchSettings
from .tracking import save_batch


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="limix-16m")
    ap.add_argument("--risk", default="균형", choices=["안전하게", "균형", "도전적으로"])
    ap.add_argument("--runs", type=int, default=12)
    ap.add_argument("--max-changes", type=int, default=2)
    ap.add_argument("--fix", action="append", default=[], help="e.g. --fix catalyst=B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args(argv)

    schema = default_schema()
    plant = FakePlant(seed=a.seed)
    history = plant.history()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = Engine(history, schema, model=a.model, context=plant.context_now(history))
    print(f"모델: {eng.model_name}" + (f"  (LimiX 사용 불가: {eng.fallback_reason})" if eng.fallback_reason else ""))
    print(f"과거 실험 {len(history)}건, 예측 범위 보정 배율 {eng.model.factor:.2f}\n")

    print("== 발견 ==")
    for f in eng.findings():
        print(f"[{f.strength}] {f.sentence}")

    print("\n== 바꿔보기: 표준 조건 ==")
    std = schema.standard_recipe()
    p = eng.predict(std)
    print(f"예상 수율 {p['predicted']}% (범위 {p['low']}–{p['high']}%)")
    curve = eng.slider_curves({**std, "catalyst": "B"})["temp"]
    shown = curve.iloc[::5]
    print("촉매 B, 온도 곡선: " + ", ".join(
        f"{v:.0f}℃ {m:.1f}%{'' if s else '(데이터 없음)'}" for v, m, s in zip(shown.value, shown["mean"], shown.supported)))

    fixed = dict(kv.split("=", 1) for kv in a.fix)
    for k, v in fixed.items():
        if schema[k].is_continuous:
            fixed[k] = float(v)
    settings = BatchSettings(risk=a.risk, n_runs=a.runs, max_changes=a.max_changes, fixed=fixed, seed=a.seed)
    batch = eng.suggest(settings)
    print(f"\n== 다음 실험 ({a.risk}, {settings.resolved_split()} = 최적화/학습/기준) ==")
    for s in batch.suggestions:
        print(f"{s.run:>2}. [{s.purpose}] {s.reason}")
    path = save_batch(batch, a.out)
    print(f"\n저장: {path}")

    # a week later: run the batch on the fake plant and compare with the saved predictions
    saved = pd.read_csv(path)
    ran = plant.run(batch.to_frame())
    _, summary = eng.this_week(saved, ran[["batch_id", "run", "yield"]])
    print(f"\n== 이번 주 == {summary['text']}, 평균 오차 {summary['mean_error_pp']:+.1f}%p, "
          f"최고 수율 {ran['yield'].max():.1f}% (과거 최고 {history['yield'].max():.1f}%)")


if __name__ == "__main__":
    main()
