"""Features + Plackett-Luce glued into a walk-forward predictor, shared by both modes.

mode "pre":  FEATURES             (form, quali form, quali x circuit pole rate)
mode "post": POST_QUALI_FEATURES  (the above + grid, grid x circuit pole rate, pit-lane start)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import features
from .config import EVAL_SEASONS, POST_QUALI_COLS, PRE_WEEKEND_COLS
from .evaluate import compare, paired_diff
from .plackett_luce import DEFAULT_MODEL_PARAMS, PlackettLuce
from .ridge import Ridge

MODES = {
    "pre": {"cols": PRE_WEEKEND_COLS, "features": features.FEATURES},
    "post": {"cols": POST_QUALI_COLS, "features": features.POST_QUALI_FEATURES},
}

PARAMS = {
    **features.DEFAULT_FEATURE_PARAMS,   # half_life, season_carry, reg_carry, shrink, circuit_shrink
    **DEFAULT_MODEL_PARAMS,              # top_k, l2, lr, epochs, train_seasons, recency_half_life
    "drop": [],                          # feature names to leave out (ablation)
    "model": "pl",                       # "pl" (listwise ranking loss) or "ridge" (pointwise squared error)
    "alpha": 10.0,                       # ridge only: L2 strength
    "ridge_target": "perf",              # ridge only: "perf" (1 = won .. 0 = last) or "points"
}
# Best settings from the first pre-weekend sweep (l2, top_k both beat the default on 169 races).
PARAMS.update({"l2": 0.1, "top_k": 2})


def make_predictor(params: dict | None = None, mode: str = "pre"):
    p = PARAMS | (params or {})
    feat_p = {k: p[k] for k in features.DEFAULT_FEATURE_PARAMS}
    model_p = {k: p[k] for k in DEFAULT_MODEL_PARAMS}
    feats = [f for f in MODES[mode]["features"] if f not in p["drop"]]

    def predict(hist: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
        both = pd.concat([hist, race], ignore_index=True)       # race rows have finish = NaN
        both = pd.concat([both, features.build(both, feat_p)], axis=1)
        train, target = both.iloc[:len(hist)], both.iloc[len(hist):]
        if p["model"] == "ridge":
            m = Ridge(feats, alpha=p["alpha"], target=p["ridge_target"], train_seasons=p["train_seasons"])
        else:
            m = PlackettLuce(feats, **model_p)
        predict.model = m.fit(train)
        return predict.model.score(target)

    return predict


def parse_set(items: list) -> dict:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        if k not in PARAMS:
            raise SystemExit(f"unknown param {k!r}; choose from {list(PARAMS)}")
        out[k] = json.loads(v) if v not in ("None", "none") else None
    return out


def p_name(params: dict) -> str:
    return "ridge" if (PARAMS | params)["model"] == "ridge" else "plackett_luce"


def cli(mode: str, baselines: dict, out_dir) -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--set", nargs="*", default=[], help='override params, e.g. half_life=10 drop=\'["track_perf"]\'')
    a.add_argument("--seasons", type=int, nargs="+", default=EVAL_SEASONS)
    a.add_argument("--holdout", action="store_true")
    a.add_argument("--show-loss", action="store_true", help="print the training-loss curve of the last fit")
    args = a.parse_args()

    from .data import load

    params = parse_set(args.set)
    pl = make_predictor(params, mode)
    summary, per_race = compare(load(), {p_name(params): pl, **baselines}, MODES[mode]["cols"],
                                seasons=args.seasons, holdout=args.holdout)
    out = Path(out_dir) / "reports"
    out.mkdir(exist_ok=True)
    tag = "holdout" if args.holdout else "walkforward"
    per_race.to_csv(out / f"pl_{tag}_per_race.csv", index=False)

    print(f"\n{mode} | {tag} | seasons {args.seasons} | params {PARAMS | params}\n")
    print(summary.sort_values("winner_mrr", ascending=False).to_string())
    print("\nwinner_mrr by season:\n")
    print(per_race.pivot_table(index="model", columns="season", values="winner_mrr").round(2).to_string())
    for base in baselines:
        d, ci = paired_diff(per_race, p_name(params), base)
        print(f"\n{p_name(params)} - {base}: winner_mrr {d:+.3f} ± {ci:.3f} (95% CI, paired over the same races)", end="")
    print("\n\nweights from the last fit (standardized features, + = helps you win):\n")
    print(pl.model.weights().round(3).to_string())
    if args.show_loss:
        h = pl.model.history
        print("\ntraining loss:", " ".join(f"{h[i]:.3f}" for i in range(0, len(h), max(1, len(h) // 15))), f"... {h[-1]:.3f}")
