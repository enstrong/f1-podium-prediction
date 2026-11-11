"""Post-quali baselines: pre-weekend info + qualifying result + final grid (after penalties).

Run: uv run python post-quali/baselines.py
"""
from pathlib import Path

import pandas as pd

from core.config import POST_QUALI_COLS
from core.evaluate import cli


def grid_order(hist: pd.DataFrame, race: pd.DataFrame):
    """Finish = start. The number every post-quali model has to beat."""
    return -race["grid"].to_numpy()


def quali_order(hist: pd.DataFrame, race: pd.DataFrame):
    """Quali pace ignoring grid penalties (falls back to grid if quali missing)."""
    return -race["quali"].fillna(race["grid"]).to_numpy()


PREDICTORS = {
    "grid": grid_order,
    "quali": quali_order,
}

if __name__ == "__main__":
    cli(PREDICTORS, POST_QUALI_COLS, Path(__file__).parent)
