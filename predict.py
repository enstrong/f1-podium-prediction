"""Live prediction for the next race -> predictions/<season>_R<round>_<circuit>_<mode>.md

    python predict.py pre                     # before FP1: next round on the calendar
    python predict.py post                    # after quali: pulls that round's qualifying from the API
    python predict.py post --grid VER=20 LEC=15   # apply grid penalties announced after quali
    python predict.py pre --round 21 --swap doohan=colapinto   # entry-list change vs last race
    python predict.py pre --model lgbm        # model 2 (LightGBM LambdaRank) -> ..._pre_lgbm.md

Then commit the file BEFORE the session starts: the commit timestamp is the proof.

Pre-weekend entry list = the drivers/teams of the last race in the data (use --swap for changes).
Post-quali grid = qualifying order unless overridden with --grid (0 = pit-lane start).
History = every race strictly before the target round, so `--round <past round>` is an honest replay.

P(win) is the model's softmax. P(podium) comes from sampling finishing orders from the
Plackett-Luce model: add Gumbel noise to every driver's score and sort - each sort is exactly
one draw from P(order). Count how often each driver lands in the top 3.
"""
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from core import data
from core.pipeline import MODES, PARAMS, make_predictor, relevant_params

ROOT = Path(__file__).parent
N_SAMPLES = 20_000


def podium_odds(scores: np.ndarray, n: int = N_SAMPLES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noisy = scores[None, :] + rng.gumbel(size=(n, len(scores)))
    top3 = np.argsort(-noisy, axis=1)[:, :3]
    return np.bincount(top3.ravel(), minlength=len(scores)) / n


def git_state() -> str:
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return h + (" + uncommitted changes" if dirty else "")
    except OSError:
        return "unknown"


def build_race(mode: str, event: pd.Series, hist: pd.DataFrame, grid_over: dict, swaps: dict) -> pd.DataFrame:
    ev = {c: event[c] for c in ["season", "round", "event", "circuit_id", "location", "country", "date", "is_sprint_weekend"]}
    if mode == "pre":
        last = hist[hist["season"] * 100 + hist["round"] == (hist["season"] * 100 + hist["round"]).max()]
        entry = last[["driver_id", "driver", "driver_name", "team_id", "team"]].copy()
        for old, new in swaps.items():
            known = hist[hist["driver_id"] == new].tail(1)
            if old not in set(entry["driver_id"]):
                raise SystemExit(f"--swap: {old!r} not in the last race's entry list")
            row = entry["driver_id"] == old
            entry.loc[row, "driver_id"] = new
            entry.loc[row, "driver"] = known["driver"].iloc[0] if len(known) else new[:3].upper()
            entry.loc[row, "driver_name"] = known["driver_name"].iloc[0] if len(known) else new
        race = entry.assign(**ev)
    else:
        q = data.qualifying(int(event["season"]), int(event["round"]))
        q["grid"] = q["quali"]
        for drv, g in grid_over.items():
            hit = (q["driver"] == drv) | (q["driver_id"] == drv)
            if not hit.any():
                raise SystemExit(f"--grid: unknown driver {drv!r}")
            q.loc[hit, "grid"] = int(g)
        q["pitlane_start"] = q["grid"] <= 0
        # re-slot the grid after penalties: penalised cars move back, others close the gaps
        q["grid"] = q["grid"].where(q["grid"] > 0, len(q) + 1).rank(method="first").astype(int)
        race = q.assign(**ev)
    return race[MODES[mode]["cols"]].sort_values("driver_id").reset_index(drop=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=list(MODES))
    a.add_argument("--season", type=int, default=data.CURRENT_SEASON)
    a.add_argument("--round", type=int, help="default: the first calendar round with no result in the data")
    a.add_argument("--grid", nargs="*", default=[], help="post only: DRV=slot overrides, e.g. VER=20 (0 = pit lane)")
    a.add_argument("--swap", nargs="*", default=[], help="pre only: old_driver_id=new_driver_id")
    a.add_argument("--model", choices=["pl", "lgbm"], default="pl", help="pl = model 1, lgbm = model 2")
    a.add_argument("--out", type=Path, default=ROOT / "predictions")
    a.add_argument("--force", action="store_true", help="overwrite an existing prediction file")
    args = a.parse_args()

    hist_all = data.load()
    cal = data.schedule(args.season)
    done = set(hist_all.loc[hist_all["season"] == args.season, "round"])
    rnd = args.round or min(set(cal["round"]) - done)
    event = cal.set_index("round").loc[rnd]
    event["round"] = rnd
    key = args.season * 100 + rnd
    hist = hist_all[hist_all["season"] * 100 + hist_all["round"] < key]

    race = build_race(args.mode, event, hist, dict(x.split("=") for x in args.grid),
                      dict(x.split("=") for x in args.swap))
    params = PARAMS | {"model": args.model}
    predict = make_predictor({"model": args.model}, args.mode)
    s = predict(hist, race)
    T = 1.0
    if args.model == "lgbm":
        # LambdaRank scores only order drivers; their scale isn't a probability. One temperature,
        # fitted on the last 30 races held out from a separate fit, turns them into P(win).
        T = predict.model.calibrate(predict.train)
        s = s / T
    p_win = np.exp(s - s.max()); p_win /= p_win.sum()
    out = race.assign(p_win=p_win, p_podium=podium_odds(s)).sort_values("p_win", ascending=False).reset_index(drop=True)

    suffix = "" if args.model == "pl" else f"_{args.model}"
    path = args.out / f"{args.season}_R{rnd:02d}_{event['circuit_id']}_{args.mode}{suffix}.md"
    if path.exists() and not args.force:
        raise SystemExit(f"{path} already exists - a committed prediction shouldn't be rewritten (use --force).")
    last = hist.sort_values(["season", "round"]).iloc[-1]
    label = {"pre": "pre-weekend (before FP1)", "post": "post-qualifying (before the race)"}[args.mode]
    lines = [
        f"# {args.season} R{rnd} {event['event']}: {label}",
        "",
        f"- Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, race date {event['date']:%Y-%m-%d}",
        f"- Data through: {int(last['season'])} R{int(last['round'])} {last['event']}",
        f"- Code: {git_state()}",
        f"- Scope: {'first model' if args.model == 'pl' else 'model 2'}, built only from previous race and qualifying results. It does NOT use "
        "weather, tyre strategy, practice pace, upgrades or many other things that decide races.",
        "- Model: " + ("model 1, Plackett-Luce" if args.model == "pl" else
                       f"model 2, LightGBM LambdaRank (P(win) = softmax(score / T), T = {T:.3f} fitted on held-out races)")
        + f", params {relevant_params(params)}",
        "",
        f"**Podium pick: {' / '.join(out['driver'][:3])}**",
        "",
        "| # | Driver | Team |" + (" Grid |" if args.mode == "post" else "") + " P(win) | P(podium) |",
        "|---|---|---|" + ("---|" if args.mode == "post" else "") + "---|---|",
    ]
    for i, r in out.iterrows():
        g = f" {int(r['grid'])}{' (pit lane)' if r['pitlane_start'] else ''} |" if args.mode == "post" else ""
        lines.append(f"| {i + 1} | {r['driver']} | {r['team']} |{g} {r['p_win']:.1%} | {r['p_podium']:.1%} |")
    text = "\n".join(lines) + "\n"
    args.out.mkdir(exist_ok=True)
    path.write_text(text)
    print(text)
    print(f"-> {path}\n   commit it before the session starts.")


if __name__ == "__main__":
    main()
