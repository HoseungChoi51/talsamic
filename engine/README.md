# yieldengine

The engine behind the yield optimization tool for plant engineers. The UI and the
LLM both call this package, so the LLM is optional and the UI can be replaced.

**The model finds the structure and the engineers judge it.** The engine runs a
fixed set of analyses and states each result as a plain Korean sentence. It also
suggests runs. Engineers only react to findings and adjust plain-language controls.

Everything runs on a fake plant (`simulator.py`). The yield formula is made up and
is not real chemistry.

## Screens → functions

| Screen | Call | Module |
|---|---|---|
| 이번 주 | `Engine.this_week(saved_batch, actuals)` → "10/12건 예측 범위 안" | `tracking.py` |
| 데이터 정리 | `Engine.suspect_values()`, `ColumnMapping`, `convert_units` | `cleaning.py` |
| 발견 | `Engine.findings()` → `Finding(sentence, strength, facts, chart)`, `meeting_agenda` | `analysis/` |
| 바꿔보기 | `Engine.predict(recipe)`, `Engine.slider_curves(recipe)` | `whatif.py` |
| 다음 실험 | `Engine.suggest(BatchSettings(...))` | `suggest.py` |

```python
from yieldengine.engine import Engine
from yieldengine.simulator import FakePlant, default_schema
from yieldengine.suggest import BatchSettings

schema = default_schema()
plant = FakePlant(seed=0)
history = plant.history()                      # 140 past runs, 28 weeks
eng = Engine(history, schema, context=plant.context_now(history))

for f in eng.findings():
    print(f.strength, f.sentence)

batch = eng.suggest(BatchSettings(risk="안전하게", n_runs=12, fixed={"catalyst": "B"},
                                  limits={"temp": (150, 190)}, max_changes=2))
```

Run the whole loop: `python -m yieldengine.demo` (options: `--model gp`,
`--risk 도전적으로`, `--fix catalyst=B`, `--max-changes 1`). The demo prints the
findings, a slider curve, a 12-run batch, and then runs the batch on the fake plant
and compares the results with the saved predictions.

## Models

`model_factory(schema, "limix-16m")` is the default. If LimiX's code or weights
aren't available, it falls back to the GP and gives the reason in
`factory.fallback_reason`. The demo prints that reason.

- **LimiX-16M** (`models/limix.py`). This is an in-context model: past runs are the
  context and candidates are the queries, scored in one forward pass. LimiX returns
  point predictions for regression. The range combines the spread from bootstrap
  resamples of the context with an out-of-fold error floor.
  Setup:
  ```bash
  git clone https://github.com/limix-ldm-ai/LimiX && pip install -e LimiX
  export YIELDENGINE_LIMIX_CKPT=/path/to/LimiX-16M.ckpt   # else fetched from HF stable-ai/LimiX-1_16M
  export YIELDENGINE_LIMIX_CONFIG=/path/to/LimiX/config/reg_default_noretrieval.json
  ```
- **GP** (`models/gp.py`). This is the baseline: a Matern ARD kernel with a noise
  term. It always runs.
- Adding a model only requires `fit(df)` and `predict(df) -> Prediction(mean, sd)`.

**Calibration** (`calibration.py`). The engine refits on earlier runs and predicts
later blocks (rolling origin). If the 80% ranges cover less than 80% of those runs,
every range is widened by the same factor. Ranges are never narrowed.

**New-chemical guard** (`novelty.py`). A model learns what a ligand, base or catalyst
does only from runs that used it, so a recipe with a value the past runs never used is
a guess. In the benchmarks the foundation models were overconfident there: with
held-out ligands, their 80% ranges covered as little as 13% of the actual yields. The
`Engine` wraps every model in `NoveltyGuard`, which

- lists the new conditions per row in `Prediction.novel`, `predict_recipe(...)["new_conditions"]`
  and `Suggestion.facts["new_conditions"]`, and adds a sentence to the suggestion's reason;
- widens those rows' ranges to at least the error measured by holding out one value at
  a time from the past runs (leave-one-value-out). With fewer than two values tried, it
  uses the spread of all past yields;
- leaves the predicted mean alone. Only set-before conditions are checked; a new
  raw-material lot is the drift analysis's job.

Benchmarks take `--guard` to compare with and without it.

## 발견: what is computed

| Finding | How | Example sentence (fake data) |
|---|---|---|
| Aliasing | Inputs with \|Spearman\| ≥ 0.9 become one group | 압력과 가스 유량은 지금까지 항상 같이 움직여서 (상관 0.95) 어느 쪽이 수율을 바꾸는지 구분할 수 없다. |
| Importance | Grouped permutation: each group is shuffled jointly, and the result is how much predicted yield moves, in %p | 촉매는 수율에 영향이 크다: 이 조건만 섞어 보면 예측 수율이 평균 1.3%p 움직인다. |
| No effect | Importance share under 3% | 교반 속도는 지금까지 써 본 범위(250 rpm–520 rpm)에서는 수율과 거의 관계없다. |
| Interaction | PD per category; one category's effect must be at least twice the others | 반응 온도 효과는 촉매 B에서만 크다 (B: 5.6%p, A·C: 0.5%p 이하). |
| Sweet spot / edge | PD peak within the range that has data. Grid points with fewer than 3 past runs nearby are ignored and marked `supported=False` | 촉매 B에서 반응 온도를 높일수록 수율이 오르고, 데이터가 있는 끝(184℃)까지 계속 오른다. 그 너머는 아직 모른다. |
| Drift | Out-of-time errors from the last 2 weeks, plus the weekly reference runs from the last 3 weeks | 기준 실험 수율이 최근 3주 평균 74.7%로 이전(75.8%)보다 1.2%p 낮다, … |

**Strength label**: each finding is recomputed on the earlier half and the later
half of the runs, each with its own refitted model. If it holds in both halves the
label is 근거 강함, in one half 보통, and in neither 약함 (hidden by default).
Drift uses two independent signals instead: if both agree the label is 근거 강함,
and if only one fires it is 보통.

Every number in a sentence is also in `finding.facts`. A test enforces this, so the
LLM guard can check any rephrasing.

## 다음 실험: controls

| Control | Field | Effect |
|---|---|---|
| 모험 정도 | `risk`: 안전하게 / 균형 / 도전적으로 | Ranks by the 25th percentile / expected improvement / the 90th percentile. The maximum distance from the nearest past run is 0.15 / 0.30 / 0.60 (1 = one condition moved across its whole range) |
| 실험 수, split | `n_runs`, `split` | The default is about 3:1 optimize:learn plus one reference run, e.g. 8/3/1 for 12 |
| 고정할 조건 | `fixed` | Fixed conditions are always applied. They don't count toward the change limit |
| 범위 제한 | `limits` | `(lo, hi)` for numbers, or a list of allowed categories |
| 한 번에 바꿀 조건 수 | `max_changes` | Maximum number of changes from the standard recipe |
| 새 값 시도 | `max_new` | How many optimize runs may use a value never run before (a new ligand, base...). Set by the risk level: 0 / 1 / no limit. Fixed conditions don't count. Learn runs aren't limited, so new values get tried there |

Learn runs come first from aliased groups: one condition is moved while its partner
stays at the standard value. The remaining learn runs go to the most uncertain
candidates nearby. The reference run is the standard recipe. If the constraints
leave room for fewer distinct runs than requested, `batch.notes` says so.

## What the LLM may do

The LLM may call the read-only methods plus one "propose settings" tool that
returns `BatchSettings`, which take effect only after an engineer clicks. It never
calls `suggest` to pick runs. Any text it writes passes through
`explain.checked_reason(text, facts, template)`: if the text contains a number that
isn't in `facts`, the fixed template is shown instead.

## Development

```bash
pip install -e ".[dev]"
pytest -q            # ~10 s, GP only
```

## Benchmarks

There are three real reaction-yield datasets, downloaded from GitHub and cached in
`~/.cache/yieldengine`. They're harder than the fake plant because the conditions are
chemicals: a tabular model sees only category codes, while an LLM knows what the
chemicals are.

| Benchmark | Rows | Conditions | Hard splits |
|---|---|---|---|
| `buchwald_hartwig` (Ahneman 2018) | 3955 | ligand, base, aryl halide, additive | `additive_test1-4` (the paper's held-out additives) |
| `suzuki` (Perera 2018) | 5760 | electrophile, nucleophile, ligand, base, solvent | `ligand_out1-4` |
| `direct_arylation` (Shields 2021, EDBO) | 1728 | ligand, base, solvent, concentration, temperature | `ligand_out1-4` |

Every benchmark also has `random70`, plus `n100`/`n100b`: 100 training runs, about
the size of a plant's history.

```bash
# prediction: accuracy, and whether the 80% ranges hold
python -m yieldengine.benchmarks.run --models gp exaone-tabular tabfm --out results/prediction.csv
# optimization: start from 10 low-yield runs, then 8 rounds of 8 suggestions (6 optimize + 2 learn)
python -m yieldengine.benchmarks.campaign --models random gp exaone-tabular --start low --seeds 0 1 2 3
```

The GP baseline is limited to 500 training rows, because exact GP gets slow fast.
The prediction results split coverage into test rows with a value the training rows
never used (`coverage80_new`) and the rest (`coverage80_seen`).
