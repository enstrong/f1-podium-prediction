"""One-knob-at-a-time sweep around the default PARAMS in core/pipeline.py.

Run:   python -m core.sweep --mode pre                       # all knobs below, all eval seasons
       python -m core.sweep --mode post --only l2 circuit_shrink
       python -m core.sweep --mode pre --seasons 2025 2026   # faster, noisier

Each row is scored against the default config on the SAME races (paired difference),
which is far less noisy than comparing two means. If the CI crosses 0, call it a tie.
Results -> <mode folder>/reports/sweep_<seasons>.csv
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from core.config import EVAL_SEASONS
from core.evaluate import bootstrap_ci
from core.pipeline import MODES, PARAMS

GRID = {
    # log scales for strengths, like the lambda lesson from the NCF project
    "l2": [0.001, 0.01, 0.1, 1.0, 10.0],
    "half_life": [2, 4, 10, 20],
    "season_carry": [0.3, 0.5, 0.9, 1.0],
    "reg_carry": [0.05, 0.5, 1.0],          # 1.0 = "ignore regulation changes"
    "shrink": [0.1, 3.0, 10.0],
    "top_k": [1, 2, 3, 5],
    "train_seasons": [1, 2, 4],
    "recency_half_life": [1, 3],
    "circuit_shrink": [1.0, 3.0, 30.0, 1000.0],
    "model": ["ridge"],
    "half_life_fast": [1, 3, 4],
    "track_shrink": [0.5, 5.0, 20.0],
    "chaos_form": [0.5, 0.9],
    "chaos_loss": [0.5, 0.9],  # 1000 = "every circuit is the same"
}
FOLDER = {"pre": "pre-weekend", "post": "post-quali"}


def run(args):
    name, override, seasons, mode = args
    import torch
    torch.set_num_threads(1)
    from core.data import load
    from core.evaluate import walk_forward
    from core.pipeline import make_predictor
    df = walk_forward(load(), make_predictor(override, mode), MODES[mode]["cols"], seasons=seasons)
    return name, override, df


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--mode", choices=list(MODES), default="pre")
    a.add_argument("--only", nargs="*", default=list(GRID) + ["drop"])
    a.add_argument("--seasons", type=int, nargs="+", default=EVAL_SEASONS)
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = a.parse_args()

    grid = GRID | {"drop": [[f] for f in MODES[args.mode]["features"]]}
    jobs = [("default", {}, args.seasons, args.mode)]
    for k in args.only:
        jobs += [(f"{k}={v}", {k: v}, args.seasons, args.mode) for v in grid[k] if v != PARAMS.get(k)]
    tag = "all" if args.seasons == EVAL_SEASONS else "_".join(map(str, args.seasons))
    print(f"{len(jobs)} configs x walk-forward over seasons {args.seasons} on {args.workers} workers...")

    results = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for name, override, df in ex.map(run, jobs):
            results[name] = df
            print(f"  done {name}", flush=True)

    base = results["default"].set_index(["season", "round"])
    rows = []
    for name, df in results.items():
        d = df.set_index(["season", "round"])
        diff = d["winner_mrr"] - base["winner_mrr"]
        rows.append({"config": name, "winner_mrr": d["winner_mrr"].mean(), "podium_p@3": d["podium_p@3"].mean(),
                     "ndcg@3": d["ndcg@3"].mean(), "Δmrr_vs_default": diff.mean(),
                     "Δ_ci95": bootstrap_ci(diff) if name != "default" else 0.0})
    out = pd.DataFrame(rows).round(3)
    path = Path(__file__).parent.parent / FOLDER[args.mode] / "reports" / f"sweep_{tag}.csv"
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path, index=False)
    out["verdict"] = [("better" if r["Δmrr_vs_default"] - r["Δ_ci95"] > 0 else
                       "worse" if r["Δmrr_vs_default"] + r["Δ_ci95"] < 0 else "tie") if r["config"] != "default" else ""
                      for _, r in out.iterrows()]
    print("\n" + out.to_string(index=False))
    print(f"\n-> {path}")
