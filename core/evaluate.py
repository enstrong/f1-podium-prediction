"""Walk-forward evaluation: for each race, train/predict on everything strictly before it.

A predictor is `f(history, race) -> scores` (higher = better), one score per row of `race`.
`race` only contains the columns allowed for that mode (see config), so it can't leak results.
"""
from typing import Callable

import numpy as np
import pandas as pd

from .config import EVAL_SEASONS, HOLDOUT
from .metrics import METRICS, race_metrics

Predictor = Callable[[pd.DataFrame, pd.DataFrame], "np.ndarray | pd.Series"]


def _key(df: pd.DataFrame) -> pd.Series:
    return df["season"] * 100 + df["round"]


def is_holdout(season: int, event: str) -> bool:
    return event in HOLDOUT.get(season, [])


def walk_forward(results: pd.DataFrame, predictor: Predictor, visible_cols: list,
                 seasons=EVAL_SEASONS, holdout: bool = False) -> pd.DataFrame:
    results = results.assign(key=_key(results))
    races = results[["season", "round", "event", "key"]].drop_duplicates().sort_values("key")
    rows = []
    for r in races.itertuples():
        if r.season not in seasons or is_holdout(r.season, r.event) != holdout:
            continue
        hist = results[results["key"] < r.key].drop(columns="key")
        full = results[results["key"] == r.key]
        # sort by id so ties can't fall back to finishing order
        race = full[visible_cols].sort_values("driver_id").reset_index(drop=True)
        scores = np.asarray(predictor(hist, race), dtype=float)
        pred = race.assign(_s=scores).sort_values("_s", ascending=False, kind="stable")["driver_id"].tolist()
        actual = full.sort_values("finish")["driver_id"].tolist()
        rows.append({"season": r.season, "round": r.round, "event": r.event, **race_metrics(pred, actual)})
    return pd.DataFrame(rows)


def compare(results: pd.DataFrame, predictors: dict, visible_cols: list, **kw) -> pd.DataFrame:
    """Mean of every metric for each predictor. Returns (summary, per_race)."""
    per_race, summary = [], {}
    for name, f in predictors.items():
        df = walk_forward(results, f, visible_cols, **kw).assign(model=name)
        if df.empty:
            raise SystemExit(f"No races to score for {kw}. Is the data pulled? Seasons in data: "
                             f"{sorted(results['season'].unique())}")
        per_race.append(df)
        summary[name] = df[METRICS].mean().round(3).to_dict() | {"races": len(df)}
    return pd.DataFrame(summary).T, pd.concat(per_race, ignore_index=True)


def cli(predictors: dict, visible_cols: list, out_dir) -> None:
    """Shared entry point for the per-mode scripts."""
    import argparse
    from pathlib import Path

    from .data import load

    p = argparse.ArgumentParser()
    p.add_argument("--holdout", action="store_true", help="score the final holdout races (do this ONCE)")
    p.add_argument("--seasons", type=int, nargs="+", default=EVAL_SEASONS)
    a = p.parse_args()

    summary, per_race = compare(load(), predictors, visible_cols, seasons=a.seasons, holdout=a.holdout)
    tag = "holdout" if a.holdout else "walkforward"
    out = Path(out_dir) / "reports"
    out.mkdir(exist_ok=True)
    per_race.to_csv(out / f"{tag}_per_race.csv", index=False)
    summary.to_csv(out / f"{tag}_summary.csv")
    print(f"\n{tag} | seasons {a.seasons}\n")
    print(summary.sort_values("winner_mrr", ascending=False).to_string())
