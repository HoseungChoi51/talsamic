"""이번 주: predictions are saved when a batch is exported, then compared with what happened.

The "10/12건 예측 범위 안" stat is how the engineers learn whether to trust the model,
and it keeps checking the calibration step.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .suggest import Batch


def save_batch(batch: Batch, folder: str | Path) -> Path:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"batch_{batch.created_at[:10]}_{batch.batch_id}"
    batch.to_frame().to_csv(folder / f"{stem}.csv", index=False, encoding="utf-8-sig")  # opens in Excel
    (folder / f"{stem}.json").write_text(json.dumps(batch.to_dict(), ensure_ascii=False, indent=2, default=str),
                                         encoding="utf-8")
    return folder / f"{stem}.csv"


def compare_with_actuals(saved: pd.DataFrame, actuals: pd.DataFrame, target: str = "yield") -> tuple[pd.DataFrame, dict]:
    """``saved`` is the exported batch CSV; ``actuals`` has ``batch_id``, ``run`` and the target."""
    key = lambda df: df.assign(batch_id=df["batch_id"].astype(str), run=df["run"].astype(int))  # noqa: E731
    m = key(saved).merge(key(actuals[["batch_id", "run", target]]), on=["batch_id", "run"], how="inner")
    m["error"] = m[target] - m["predicted"]
    m["in_range"] = (m[target] >= m["low"]) & (m[target] <= m["high"])
    n, k = len(m), int(m["in_range"].sum())
    summary = {"total": n, "in_range": k, "text": f"{k}/{n}건 예측 범위 안",
               "mean_error_pp": round(float(m["error"].mean()), 1) if n else None,
               "by_purpose": m.groupby("purpose")["in_range"].agg(["sum", "count"]).to_dict("index")}
    return m, summary
