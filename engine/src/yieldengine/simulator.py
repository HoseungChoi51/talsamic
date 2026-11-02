"""A fake plant for demos and tests. Nothing here is real chemistry.

The made-up yield formula plants the structures the 발견 screen should find:

* 온도 has a sweet spot, but only with 촉매 B (interaction).
* 가스 유량 matters and 압력 does not, yet operators always moved them together,
  so history alone cannot tell them apart (aliasing).
* 반응 시간 helps with diminishing returns; 교반 속도 does nothing.
* 원료 lot shifts yield a little; the last three weeks lose about 2%p (drift).
* 최고 온도 is measured during the run, so it must never be a model input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import Kind, Role, Schema, Variable

START = pd.Timestamp("2026-03-02")
LOTS = [f"L{i}" for i in range(1, 11)]


def default_schema(horizon_days: int = 400) -> Schema:
    C, K = Kind.CONTINUOUS, Kind.CATEGORICAL
    return Schema(
        [
            Variable("temp", "반응 온도", C, Role.SET_BEFORE, 150, 200, standard=170, unit="℃", step=1),
            Variable("pressure", "압력", C, Role.SET_BEFORE, 1.0, 3.0, standard=2.0, unit=" bar", step=0.1),
            Variable("gas_flow", "가스 유량", C, Role.SET_BEFORE, 10, 30, standard=20, unit=" L/min", step=0.5),
            Variable("time", "반응 시간", C, Role.SET_BEFORE, 60, 180, standard=120, unit="분", step=5),
            Variable("stir", "교반 속도", C, Role.SET_BEFORE, 200, 600, standard=400, unit=" rpm", step=10),
            Variable("catalyst", "촉매", K, Role.SET_BEFORE, categories=["A", "B", "C"], standard="A"),
            Variable("lot", "원료 lot", K, Role.CONTEXT, categories=LOTS),
            Variable("days", "경과일", C, Role.CONTEXT, 0, horizon_days, unit="일", step=1),
            Variable("peak_temp", "최고 온도", C, Role.MEASURED, 150, 215, unit="℃"),
            Variable("yield", "수율", C, Role.TARGET, 0, 100, unit="%", step=0.1),
        ]
    )


@dataclass
class FakePlant:
    seed: int = 0
    noise_sd: float = 0.8
    drift_start_day: float = 175.0  # fouling begins three weeks before the end of history
    drift_size: float = 2.0
    lot_effect: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed + 1000)
        if not self.lot_effect:
            self.lot_effect = {lot: float(e) for lot, e in zip(LOTS, rng.normal(0, 0.6, len(LOTS)))}
        self.rng = np.random.default_rng(self.seed)

    # -- the hidden truth ---------------------------------------------------
    def true_yield(self, df: pd.DataFrame) -> np.ndarray:
        T = df["temp"].to_numpy(float)
        g = df["gas_flow"].to_numpy(float)
        t = df["time"].to_numpy(float)
        cat = df["catalyst"].astype(str).to_numpy()
        days = df["days"].to_numpy(float)
        y = 70.0
        y = y + np.select([cat == "B", cat == "C"], [2.5, -1.0], 0.0)
        y = y - 0.004 * (T - 172) ** 2
        y = y + np.where(cat == "B", 6.0 * np.exp(-(((T - 185) / 9) ** 2)), 0.0)
        y = y + 5.0 * (1 - np.exp(-(g - 10) / 7))
        y = y + 2.5 * (1 - np.exp(-(t - 60) / 35))
        y = y + df["lot"].astype(str).map(self.lot_effect).fillna(0).to_numpy(float)
        y = y - self.drift_size * np.clip((days - self.drift_start_day) / 7, 0, 1)
        return np.clip(y, 0, 100)

    def run(self, recipes: pd.DataFrame) -> pd.DataFrame:
        """Run recipes (must include lot and days) and return them with measured columns."""
        out = recipes.copy()
        n = len(out)
        out["yield"] = np.round(self.true_yield(out) + self.rng.normal(0, self.noise_sd, n), 1)
        cat_b = (out["catalyst"].astype(str) == "B").to_numpy()
        out["peak_temp"] = np.round(out["temp"].to_numpy(float) + 3 + 2 * cat_b + self.rng.normal(0, 1, n), 1)
        if "date" not in out:
            out["date"] = START + pd.to_timedelta(out["days"], unit="D")
        return out

    # -- history as operators actually ran the plant ------------------------
    def history(self, weeks: int = 28, runs_per_week: int = 5) -> pd.DataFrame:
        rng = self.rng
        rows = []
        for w in range(weeks):
            lot = LOTS[min(w // 4, len(LOTS) - 1)]
            for k in range(runs_per_week):
                day = w * 7 + k
                if k == 0:  # weekly reference run at the standard recipe
                    rows.append(dict(temp=170, pressure=2.0, gas_flow=20.0, time=120, stir=400,
                                     catalyst="A", lot=lot, days=day, is_reference=True))
                    continue
                cat = rng.choice(["A", "B", "C"], p=[0.6, 0.25, 0.15])
                temp = rng.uniform(160, 184) if cat == "B" else rng.normal(170, 5)
                pressure = float(np.clip(rng.normal(2.0, 0.28), 1.2, 2.8))
                gas = 10 * pressure + rng.normal(0, 0.5)  # habit: always moved together
                rows.append(dict(
                    temp=round(float(np.clip(temp, 155, 186))),
                    pressure=round(pressure, 1),
                    gas_flow=round(float(np.clip(gas, 10, 30)) * 2) / 2,
                    time=int(rng.choice([90, 120, 150], p=[0.25, 0.5, 0.25])),
                    stir=int(round(np.clip(rng.normal(400, 60), 250, 550), -1)),
                    catalyst=str(cat), lot=lot, days=day, is_reference=False,
                ))
        df = pd.DataFrame(rows)
        df["date"] = START + pd.to_timedelta(df["days"], unit="D")
        return self.run(df)

    def context_now(self, history: pd.DataFrame) -> dict:
        """Context for the next batch: the current lot, one day after the last run."""
        last = history.sort_values("days").iloc[-1]
        return {"lot": last["lot"], "days": float(last["days"]) + 1}
