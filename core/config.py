from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / "cache"
RESULTS_PATH = DATA_DIR / "results.parquet"

FIRST_SEASON = 2018
CURRENT_SEASON = 2026

# Seasons whose races get scored in walk-forward tuning (history always uses everything earlier).
# 2018 is history-only: there's nothing before it to learn from.
EVAL_SEASONS = list(range(FIRST_SEASON + 1, CURRENT_SEASON + 1))

# Major regulation resets: last season's form says much less about these seasons.
REG_CHANGE_SEASONS = [2022, 2026]

# Final holdout: scored ONCE at the end with --holdout. Never tune against these.
# Matched on circuit_id (event names change year to year: "São Paulo" vs "Brazilian" GP etc.).
HOLDOUT = {
    2026: ["americas", "rodriguez", "interlagos"],  # US, Mexico City, Brazil
}

# Columns a predictor is allowed to see for the race it's predicting. Everything else
# (finish, points, status, ...) is hidden by the evaluator, so leakage is impossible.
EVENT_COLS = ["season", "round", "event", "circuit_id", "location", "country", "date", "is_sprint_weekend"]
ENTRY_COLS = ["driver_id", "driver", "driver_name", "team_id", "team"]
PRE_WEEKEND_COLS = EVENT_COLS + ENTRY_COLS
POST_QUALI_COLS = PRE_WEEKEND_COLS + ["quali", "q1", "q2", "q3", "grid", "pitlane_start"]
