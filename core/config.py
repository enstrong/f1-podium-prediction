from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_PATH = DATA_DIR / "results.parquet"

FIRST_SEASON = 2018
CURRENT_SEASON = 2026

# Seasons whose races get scored in walk-forward tuning (history always uses everything earlier).
EVAL_SEASONS = [2025, 2026]

# Final holdout: scored ONCE at the end with --holdout. Never tune against these.
# Matched against FastF1's EventName.
HOLDOUT = {
    2026: ["United States Grand Prix", "Mexico City Grand Prix", "São Paulo Grand Prix"],
}

# Columns a predictor is allowed to see for the race it's predicting. Everything else
# (finish, points, status, ...) is hidden by the evaluator, so leakage is impossible.
EVENT_COLS = ["season", "round", "event", "location", "country", "date", "is_sprint_weekend"]
ENTRY_COLS = ["driver_id", "driver", "driver_name", "team_id", "team"]
PRE_WEEKEND_COLS = EVENT_COLS + ENTRY_COLS
POST_QUALI_COLS = PRE_WEEKEND_COLS + ["quali", "grid", "pitlane_start"]
