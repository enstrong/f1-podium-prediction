import math

import numpy as np
import pandas as pd
import pytest
import torch

from core import features
from core.plackett_luce import PlackettLuce, pl_loss
from tests.test_core import fake_results


def test_pl_loss_two_drivers_is_logistic():
    # with two drivers, P(a beats b) = sigmoid(s_a - s_b)
    s = torch.tensor([[2.0, 0.5]])
    rank = torch.tensor([[0, 1]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert pl_loss(s, rank, mask, top_k=1).item() == pytest.approx(math.log(1 + math.exp(-1.5)), rel=1e-5)


def test_pl_loss_ignores_padding_and_constant_shift():
    s = torch.tensor([[1.0, 0.0, -1.0, 99.0]])
    rank = torch.tensor([[0, 1, 2, 10_000]])
    mask = torch.tensor([[True, True, True, False]])
    a = pl_loss(s, rank, mask, top_k=3)
    b = pl_loss(s + 5, rank, mask, top_k=3)
    assert a.item() == pytest.approx(b.item(), rel=1e-5)
    # top-3 of 3 drivers: last factor is P(only one left) = 1 -> log 1 = 0
    assert pl_loss(s, rank, mask, top_k=2).item() == pytest.approx(a.item(), rel=1e-5)


def test_features_are_causal():
    res = fake_results(seasons=(2021, 2022, 2023), rounds=5)
    f1 = features.build(res)
    # scramble everything from 2023 R3 onwards: features up to and including R3 must not change
    later = (res["season"] * 100 + res["round"]) > 202303
    res2 = res.copy()
    res2.loc[later, "finish"] = res2.loc[later, "finish"].sample(frac=1, random_state=0).to_numpy()
    f2 = features.build(res2)
    upto = (res["season"] * 100 + res["round"]) <= 202303
    pd.testing.assert_frame_equal(f1[upto], f2[upto])


def test_reg_change_discounts_old_seasons():
    res = fake_results(seasons=(2021, 2022), rounds=3)
    first_2022 = (res["season"] == 2022) & (res["round"] == 1)
    weak = features.build(res, {"reg_carry": 0.01})[first_2022]["evidence"].mean()
    strong = features.build(res, {"reg_carry": 1.0})[first_2022]["evidence"].mean()
    assert weak < strong


def test_model_learns_obvious_signal():
    # finish is driven by "team_perf"-like signal: a model on good features should rank well
    res = fake_results(seasons=(2022, 2023), rounds=8, n=10, seed=3)
    F = features.build(res)
    df = pd.concat([res, F], axis=1)
    m = PlackettLuce(features.FEATURES, epochs=200).fit(df)
    assert m.history[-1] < m.history[0]
    assert np.isfinite(m.weights()).all()


def test_circuit_pole_rate_is_causal_and_shrunk():
    races = pd.DataFrame({"circuit_id": ["a", "b", "a", "a"]})
    pole_won = np.array([1.0, 0.0, 1.0, np.nan])           # last race not run yet
    r = features.circuit_pole_rate(races, pole_won, k=2.0)
    assert r[0] == features.GLOBAL_POLE_WIN_PRIOR          # nothing seen yet
    # race 2 (circuit a): 1 earlier race at a (won from pole), global rate over races 0-1 = 0.5
    assert r[2] == pytest.approx((1 + 2 * 0.5) / (1 + 2))
    # race 3: 2 at a, both pole wins; global = 2/3
    assert r[3] == pytest.approx((2 + 2 * 2 / 3) / (2 + 2))


def test_target_race_quali_does_not_leak_into_its_own_form():
    res = fake_results(seasons=(2022, 2023), rounds=4)
    last = (res["season"] == 2023) & (res["round"] == 4)
    a = res.copy()
    a.loc[last, "finish"] = np.nan                        # how the predictor sees the target race
    b = a.copy()
    b.loc[last, "quali"] = b.loc[last, "quali"].to_numpy()[::-1]
    fa, fb = features.build(a), features.build(b)
    pd.testing.assert_series_equal(fa.loc[last, "driver_quali"], fb.loc[last, "driver_quali"])
    assert (fa.loc[last, "grid_perf"] != fb.loc[last, "grid_perf"]).sum() == 0  # grid untouched here


def test_ridge_matches_closed_form_and_one_feature_ranks_like_the_feature():
    from core.ridge import Ridge
    res = fake_results(seasons=(2022, 2023), rounds=6, n=10, seed=5)
    df = pd.concat([res, features.build(res)], axis=1)
    m = Ridge(["grid_perf"], alpha=0.0).fit(df)
    # a single positive weight can only rescale the feature -> same ranking as the grid itself
    assert m.w[0] > 0
    race = df[(df.season == 2023) & (df["round"] == 6)]
    assert list(np.argsort(-m.score(race))) == list(np.argsort(-race["grid_perf"].to_numpy()))


def test_quali_gap_uses_deepest_session():
    df = pd.DataFrame({"season": 2025, "round": 1, "quali": [1, 2, 3],
                       "q1": [91.0, 90.0, 90.5],         # P1 cruised in Q1: must not matter
                       "q2": [90.0, 90.2, np.nan],
                       "q3": [89.0, 89.445, np.nan]})
    g = features.quali_gap(df)
    assert g[0] == 0
    assert g[1] == pytest.approx(0.5)                   # Q3 vs Q3 best
    assert g[2] == pytest.approx(100 * (90.5 / 90.0 - 1))  # Q1 exit vs Q1 best, not vs P1's slow Q1


def test_chaos_score_flags_shuffled_races_and_is_causal():
    rho = np.array([0.8, 0.7, 0.75, 0.8, 0.72, 0.78, 0.1, 0.77])
    exp = np.full_like(rho, 0.75)
    c = features.chaos_score(rho, exp)
    assert c[6] == c.max()                                   # the lottery race
    c2 = features.chaos_score(np.append(rho[:7], -1.0), np.full(8, 0.75))
    assert np.allclose(c[:7], c2[:7])                        # later races can't change earlier scores


def test_lambdarank_learns_and_calibrates():
    from core.lambdarank import LambdaRank
    res = fake_results(seasons=(2021, 2022, 2023), rounds=12, n=10, seed=7)
    df = pd.concat([res, features.build(res)], axis=1)
    m = LambdaRank(features.POST_QUALI_FEATURES, n_trees=50, min_child_samples=5).fit(df)
    race = df[(df.season == 2023) & (df["round"] == 12)]
    s = m.score(race)
    assert s.shape == (len(race),) and np.isfinite(s).all()
    assert m.weights().index[0] in ("grid_perf", "grid_x_rho", "grid_x_pole", "qgap")  # fake finishes follow the grid
    assert 0.05 <= m.calibrate(df, last_n=8) <= 20
