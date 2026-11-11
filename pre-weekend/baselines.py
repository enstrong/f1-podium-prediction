"""Pre-weekend baselines: only info available before FP1 (entry list + history).

Run: uv run python pre-weekend/baselines.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from core.config import PRE_WEEKEND_COLS
from core.evaluate import cli


def random_order(hist: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
    """Sanity floor. Seeded per race so it's reproducible."""
    rng = np.random.default_rng(int(race["season"].iloc[0]) * 100 + int(race["round"].iloc[0]))
    return rng.random(len(race))


def championship(hist: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
    """Current-season points so far (last season's at round 1). Race points only, no sprints yet."""
    season = race["season"].iloc[0]
    cur = hist[hist["season"] == season]
    if cur.empty:
        cur = hist[hist["season"] == season - 1]
    pts = cur.groupby("driver_id")["points"].sum()
    return race["driver_id"].map(pts).fillna(-1).to_numpy()


def last_race(hist: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
    """Finishing position in each driver's most recent race."""
    last = hist.sort_values(["season", "round"]).groupby("driver_id")["finish"].last()
    return -race["driver_id"].map(last).fillna(30).to_numpy()


def rolling_form(n: int = 5):
    def f(hist: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
        h = hist.sort_values(["season", "round"]).groupby("driver_id")["finish"]
        avg = h.apply(lambda s: s.tail(n).mean())
        return -race["driver_id"].map(avg).fillna(30).to_numpy()
    f.__name__ = f"rolling_form_{n}"
    return f


PREDICTORS = {
    "random": random_order,
    "championship": championship,
    "last_race": last_race,
    "rolling_form_3": rolling_form(3),
    "rolling_form_5": rolling_form(5),
}

if __name__ == "__main__":
    cli(PREDICTORS, PRE_WEEKEND_COLS, Path(__file__).parent)
