"""Pull race, qualifying and sprint results from the Ergast API (jolpica mirror).

Whole seasons are fetched in pages of 100 (~12 requests/season instead of ~50 with
per-session FastF1 loads), throttled, retried on 429, and cached as raw JSON in
cache/ergast/. Past seasons are never re-fetched; the current season is re-fetched
with --refresh.

Run:  python -m core.data            (add --refresh after each new race weekend)
"""
import argparse
import json
import logging
import time

import pandas as pd
import requests

from .config import CACHE_DIR, CURRENT_SEASON, DATA_DIR, FIRST_SEASON, RESULTS_PATH

log = logging.getLogger("data")
BASE = "https://api.jolpi.ca/ergast/f1"
RAW_DIR = CACHE_DIR / "ergast"
PAGE = 100
_session = requests.Session()


def _get(url: str, params: dict) -> dict:
    wait = 5
    for _ in range(8):
        r = _session.get(url, params=params, timeout=30)
        if r.status_code == 429:
            log.warning("rate limited, sleeping %ss", wait)
            time.sleep(wait)
            wait = min(wait * 2, 120)
            continue
        r.raise_for_status()
        time.sleep(0.35)  # stay under 4 req/s
        return r.json()["MRData"]
    raise RuntimeError(f"gave up after repeated 429s: {url}")


def fetch(season: int, endpoint: str, refresh: bool = False) -> list:
    """All races of a season for endpoint in {results, qualifying, sprint}, merged across pages."""
    path = RAW_DIR / f"{season}_{endpoint}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())

    key = {"results": "Results", "qualifying": "QualifyingResults", "sprint": "SprintResults"}[endpoint]
    races, offset, total = {}, 0, None
    while total is None or offset < total:
        md = _get(f"{BASE}/{season}/{endpoint}.json", {"limit": PAGE, "offset": offset})
        total = int(md["total"])
        for race in md["RaceTable"]["Races"]:  # a race can be split across two pages
            races.setdefault(race["round"], {**race, key: []})[key].extend(race.get(key, []))
        offset += PAGE
    out = sorted(races.values(), key=lambda r: int(r["round"]))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    log.info("%s %-10s %2d races, %4d rows", season, endpoint, len(out), total)
    return out


def _driver_rows(race: dict, key: str) -> list:
    return [(int(race["season"]), int(race["round"]), r) for r in race.get(key, [])]


def build_season(season: int, refresh: bool = False) -> pd.DataFrame:
    races = fetch(season, "results", refresh)
    quali = fetch(season, "qualifying", refresh)
    sprint = fetch(season, "sprint", refresh) if season >= 2021 else []

    rows = []
    for race in races:
        loc = race["Circuit"]["Location"]
        for r in race["Results"]:
            grid = int(r["grid"])
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "event": race["raceName"],
                "circuit_id": race["Circuit"]["circuitId"],
                "location": loc["locality"],
                "country": loc["country"],
                "date": pd.Timestamp(race["date"]),
                "driver_id": r["Driver"]["driverId"],
                "driver": r["Driver"].get("code", r["Driver"]["driverId"][:3].upper()),
                "driver_name": f'{r["Driver"]["givenName"]} {r["Driver"]["familyName"]}',
                "team_id": r["Constructor"]["constructorId"],
                "team": r["Constructor"]["name"],
                "grid_raw": grid,
                "finish": int(r["position"]),
                "classified": r["positionText"],
                "status": r["status"],
                "points": float(r["points"]),
            })
    df = pd.DataFrame(rows)

    q = pd.DataFrame([{"season": s, "round": rd, "driver_id": r["Driver"]["driverId"], "quali": int(r["position"])}
                      for race in quali for s, rd, r in _driver_rows(race, "QualifyingResults")])
    sp = pd.DataFrame([{"season": s, "round": rd, "driver_id": r["Driver"]["driverId"],
                        "sprint_finish": int(r["position"]), "sprint_points": float(r["points"])}
                       for race in sprint for s, rd, r in _driver_rows(race, "SprintResults")])
    on = ["season", "round", "driver_id"]
    df = df.merge(q, on=on, how="left") if not q.empty else df.assign(quali=float("nan"))
    if sp.empty:
        sp = pd.DataFrame(columns=on + ["sprint_finish", "sprint_points"])
    df = df.merge(sp, on=on, how="left")
    df["sprint_points"] = df["sprint_points"].fillna(0.0)
    df["is_sprint_weekend"] = df.groupby("round")["sprint_finish"].transform(lambda s: s.notna().any())

    # Grid 0 = pit-lane start -> back of the grid.
    n = df.groupby("round")["driver_id"].transform("size")
    df["pitlane_start"] = df["grid_raw"] <= 0
    df["grid"] = df["grid_raw"].where(df["grid_raw"] > 0, n)
    df["dnf"] = ~df["classified"].str.isdigit()
    return df


def validate(df: pd.DataFrame) -> None:
    per_race = df.groupby(["season", "round"])
    bad = per_race["finish"].apply(lambda s: sorted(s) != list(range(1, len(s) + 1)))
    assert not bad.any(), f"broken finishing order in {list(bad[bad].index)}"
    assert (per_race["finish"].min() == 1).all()
    missing_q = per_race["quali"].apply(lambda s: s.isna().mean())
    for (s, r), frac in missing_q[missing_q > 0.5].items():
        log.warning("%s R%s: quali missing for %.0f%% of drivers", s, r, frac * 100)


def pull(first: int = FIRST_SEASON, last: int = CURRENT_SEASON, refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    frames = [build_season(y, refresh=refresh and y == CURRENT_SEASON) for y in range(first, last + 1)]
    out = pd.concat(frames, ignore_index=True).sort_values(["season", "round", "finish"])
    validate(out)
    out.to_parquet(RESULTS_PATH, index=False)
    print(out.groupby("season").agg(races=("round", "nunique"), rows=("driver_id", "size"),
                                    sprints=("is_sprint_weekend", lambda s: s.groupby(out.loc[s.index, "round"]).first().sum())))
    print(f"\nlatest race: {out.iloc[-1][['season', 'round', 'event']].to_dict()}  ->  {RESULTS_PATH}")
    return out


def load() -> pd.DataFrame:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(f"{RESULTS_PATH} missing - run `python -m core.data` first")
    return pd.read_parquet(RESULTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=int, default=FIRST_SEASON)
    p.add_argument("--last", type=int, default=CURRENT_SEASON)
    p.add_argument("--refresh", action="store_true", help="re-fetch the current season")
    a = p.parse_args()
    pull(a.first, a.last, a.refresh)
