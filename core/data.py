"""Pull race + qualifying results from FastF1 into one tidy table.

One row per (season, round, driver). Run:  uv run python -m core.data
Re-runs are incremental; --refresh re-pulls the current season.
"""
import argparse
import logging

import fastf1
import numpy as np
import pandas as pd

from .config import CACHE_DIR, CURRENT_SEASON, DATA_DIR, FIRST_SEASON, RESULTS_PATH

log = logging.getLogger("data")


def _results(year: int, rnd: int, kind: str) -> pd.DataFrame:
    s = fastf1.get_session(year, rnd, kind)
    s.load(laps=False, telemetry=False, weather=False, messages=False)
    return s.results


def pull_event(year: int, ev) -> pd.DataFrame:
    rnd = int(ev.RoundNumber)
    race = _results(year, rnd, "R")
    if race is None or race.empty:
        raise ValueError("empty race results")

    grid = pd.to_numeric(race["GridPosition"], errors="coerce")
    classified = race["ClassifiedPosition"].astype(str)
    df = pd.DataFrame({
        "season": year,
        "round": rnd,
        "event": ev.EventName,
        "location": ev.Location,
        "country": ev.Country,
        "date": pd.Timestamp(ev.Session5DateUtc),
        "is_sprint_weekend": "sprint" in str(ev.EventFormat).lower(),
        "driver_id": race["DriverId"].values,
        "driver": race["Abbreviation"].values,
        "driver_name": race["FullName"].values,
        "team_id": race["TeamId"].values,
        "team": race["TeamName"].values,
        "grid_raw": grid.values,
        "finish": pd.to_numeric(race["Position"], errors="coerce").values,
        "classified": classified.values,
        "status": race["Status"].values,
        "points": pd.to_numeric(race["Points"], errors="coerce").fillna(0).values,
    })
    # Grid 0 / missing = pit-lane start -> put them at the back.
    df["pitlane_start"] = ~(df["grid_raw"] > 0)
    df["grid"] = df["grid_raw"].where(df["grid_raw"] > 0, len(df))
    df["dnf"] = ~df["classified"].str.isdigit()

    try:
        q = _results(year, rnd, "Q")[["DriverId", "Position"]]
        q = q.rename(columns={"DriverId": "driver_id", "Position": "quali"})
        df = df.merge(q, on="driver_id", how="left")
    except Exception as e:  # cancelled quali etc.
        log.warning("%s R%s: no quali (%s)", year, rnd, e)
        df["quali"] = np.nan
    return df


def completed_events(year: int) -> pd.DataFrame:
    sched = fastf1.get_event_schedule(year, include_testing=False)
    race_end = pd.to_datetime(sched["Session5DateUtc"]) + pd.Timedelta(hours=3)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return sched[race_end < now]


def pull(first: int = FIRST_SEASON, last: int = CURRENT_SEASON, refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    fastf1.set_log_level("WARNING")

    have = pd.read_parquet(RESULTS_PATH) if RESULTS_PATH.exists() else pd.DataFrame()
    if refresh and not have.empty:
        have = have[have["season"] != CURRENT_SEASON]
    done = set(zip(have["season"], have["round"])) if not have.empty else set()

    frames = [have] if not have.empty else []
    for year in range(first, last + 1):
        new = []
        for _, ev in completed_events(year).iterrows():
            if (year, int(ev.RoundNumber)) in done:
                continue
            try:
                new.append(pull_event(year, ev))
                log.info("%s R%02d %s ok", year, ev.RoundNumber, ev.EventName)
            except Exception as e:
                log.error("%s R%02d %s FAILED: %s", year, ev.RoundNumber, ev.EventName, e)
        if new:
            frames.extend(new)
            out = pd.concat(frames, ignore_index=True).sort_values(["season", "round", "finish"])
            out.to_parquet(RESULTS_PATH, index=False)  # checkpoint after each season
    out = pd.read_parquet(RESULTS_PATH)
    print(out.groupby("season").agg(races=("round", "nunique"), rows=("driver_id", "size")))
    return out


def load() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} missing - run `uv run python -m core.data` first")
    return pd.read_parquet(RESULTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=FIRST_SEASON)
    p.add_argument("--last", type=int, default=CURRENT_SEASON)
    p.add_argument("--refresh", action="store_true", help="re-pull the current season")
    a = p.parse_args()
    pull(a.first, a.last, a.refresh)
