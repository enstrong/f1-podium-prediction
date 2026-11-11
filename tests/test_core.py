import numpy as np
import pandas as pd
import pytest

from core.config import POST_QUALI_COLS, PRE_WEEKEND_COLS
from core.evaluate import walk_forward
from core.metrics import race_metrics


def fake_results(seasons=(2024, 2025), rounds=4, n=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in seasons:
        for r in range(1, rounds + 1):
            grid = rng.permutation(n) + 1
            finish = np.argsort(np.argsort(grid + rng.normal(0, 2, n))) + 1
            for d in range(n):
                rows.append(dict(season=s, round=r, event=f"GP{r}", circuit_id=f"c{r}", location="x", country="x",
                                 date=pd.Timestamp(s, 1, 1) + pd.Timedelta(days=7 * r), is_sprint_weekend=False,
                                 driver_id=f"d{d}", driver=f"D{d}", driver_name=f"Driver {d}",
                                 team_id=f"t{d // 2}", team=f"T{d // 2}", grid=grid[d], quali=grid[d],
                                 pitlane_start=False, finish=finish[d], points=max(0, 11 - finish[d]), sprint_points=0.0,
                                 status="Finished", classified=str(finish[d]), dnf=False))
    return pd.DataFrame(rows)


def test_metrics_perfect():
    m = race_metrics(["a", "b", "c", "d"], ["a", "b", "c", "d"])
    assert m["winner_mrr"] == 1 and m["podium_exact"] == 1 and m["ndcg@3"] == pytest.approx(1)


def test_metrics_partial():
    m = race_metrics(["d", "a", "c", "b"], ["a", "b", "c", "d"])
    assert m["winner_mrr"] == 0.5 and m["podium_p@3"] == pytest.approx(2 / 3) and m["podium_set"] == 0


def test_no_leakage_and_history_is_strictly_past():
    res = fake_results()

    def spy(hist, race):
        assert set(race.columns) == set(PRE_WEEKEND_COLS)
        k = race["season"].iloc[0] * 100 + race["round"].iloc[0]
        assert ((hist["season"] * 100 + hist["round"]) < k).all()
        return np.zeros(len(race))

    out = walk_forward(res, spy, PRE_WEEKEND_COLS, seasons=[2025])
    assert len(out) == 4


def test_ties_dont_use_finishing_order():
    # all-zero scores must NOT give a perfect prediction via sort order
    res = fake_results(seed=1)
    out = walk_forward(res, lambda h, r: np.zeros(len(r)), PRE_WEEKEND_COLS, seasons=[2025])
    assert out["podium_exact"].mean() < 1


def test_grid_beats_random():
    res = fake_results(rounds=10, n=20)
    grid = walk_forward(res, lambda h, r: -r["grid"].to_numpy(), POST_QUALI_COLS, seasons=[2025])
    rnd = walk_forward(res, lambda h, r: np.random.default_rng(0).random(len(r)), POST_QUALI_COLS, seasons=[2025])
    assert grid["winner_mrr"].mean() > rnd["winner_mrr"].mean()
