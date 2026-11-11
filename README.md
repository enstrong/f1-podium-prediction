# F1 podium prediction

Predict the winner and podium of F1 Grands Prix, with two separate models:

| Model | Knows | Use |
|---|---|---|
| `pre-weekend/` | entry list + all past results | posted before FP1 |
| `post-quali/` | the above + qualifying + final grid | posted after quali, before the race |

## Layout

```
core/            shared: config, FastF1 data pull, metrics, walk-forward evaluator
pre-weekend/     pre-weekend models + baselines   (reports/ is gitignored)
post-quali/      post-quali models + baselines
predictions/     live predictions, committed BEFORE the session (git timestamp = proof)
notebooks/       exploration only, real logic lives in .py
tests/
```

## Setup (conda)

```bash
conda create -n f1 python=3.12 -y && conda activate f1
pip install -e . pytest jupyterlab   # includes torch
python -m core.data              # 2018-2026 race/quali/sprint results -> data/results.parquet (~2 min, cached)
python -m core.data --refresh    # after each new race weekend
pytest
```

## Evaluate

```bash
python pre-weekend/baselines.py
python post-quali/baselines.py
python post-quali/baselines.py --seasons 2023 2024 2025 2026
python pre-weekend/plackett_luce.py                         # first real model, vs the baselines
python pre-weekend/plackett_luce.py --set half_life=10 l2=0.1 --show-loss
python post-quali/plackett_luce.py                          # same model + grid features, vs grid/quali
python -m core.sweep --mode pre --only l2 reg_carry         # one-knob-at-a-time sweep -> <mode>/reports/sweep_<seasons>.csv
```

Every summary has `mrr_ci95`: the 95% bootstrap CI half-width of `winner_mrr`. Two models whose
difference is inside that are a tie. Use the **paired** difference (printed by `plackett_luce.py`
and `core.sweep`) to compare models: it's scored on the same races, so it's ~2x tighter.

**Walk-forward:** every race in `EVAL_SEASONS` (2019-2026, ~170 races; 2018 is history only) is predicted using only races strictly before it.
The evaluator passes the predictor only the columns allowed for its mode
(`PRE_WEEKEND_COLS` / `POST_QUALI_COLS` in `core/config.py`), so it can't see results it shouldn't.

**Holdout:** 2026 US, Mexico City, São Paulo GPs are excluded from tuning. Score them once at the end:
`... baselines.py --holdout`. Then the real test is Las Vegas, Qatar, Abu Dhabi.

## Live predictions

```bash
python predict.py pre                         # next race, before FP1 (entry list = last race's)
python predict.py post                        # after quali (fetches that round's qualifying)
python predict.py post --grid VER=20 LEC=15   # apply grid penalties announced after quali (0 = pit lane)
```

Writes `predictions/<season>_R<round>_<circuit>_<mode>.md` with P(win) and P(podium) per driver,
the data cutoff and the code commit. Commit the file before the session: the timestamp is the proof.
P(podium) samples finishing orders from the Plackett-Luce model (Gumbel noise + sort).

## Metrics

- `winner_mrr` – 1 / rank of the actual winner in our order (main winner metric)
- `podium_p@3` – share of the real top 3 we named in our top 3 (main podium metric)
- `podium_set` / `podium_exact` – all three right (any order / exact order)
- `ndcg@3` – order-aware podium score (P1 weighted 3, P2 2, P3 1)

## Plackett-Luce model (both modes, `core/pipeline.py`)

`core/features.py` builds causal form features (driver and team results, podium and win rates, DNF rate,
track history, and how much evidence there is) as weighted averages over past races. The weights fade with
`half_life` (in races), are multiplied by `season_carry` at each season boundary and by `reg_carry` at each
regulation reset (`REG_CHANGE_SEASONS = [2022, 2026]`), then are shrunk towards a neutral prior by `shrink`.

Qualifying form uses quali **positions**, not lap times, so a session that turns wet in Q3 can't make
a Q1 exit look faster than pole. `circuit_pole_rate` is the share of past races at that circuit won from
pole, shrunk towards the all-circuit rate by `circuit_shrink` (with ~8 races per circuit, raw rates are mostly noise).
It's the same for every driver in a race, so it only enters through interactions: `quali_x_pole` (pre-weekend)
and `grid_x_pole` (post-quali). These let the weight on quali form or the grid be bigger at Monaco than at Monza.

`driver_qgap` / `qgap` measure **how far off** in qualifying: the % gap to the fastest lap of the deepest
session each driver reached (Q3 vs Q3, Q2 exits vs Q2, Q1 exits vs Q1). Track evolution and top cars
cruising in Q1 make cross-session comparisons misleading. 
Circuit character: `circuit_rho` is the (shrunk) average grid-vs-finish rank correlation at that circuit,
so it covers the whole order, not just pole (Monaco ~0.68, Interlagos ~0.57). It enters via `quali_x_rho` / `grid_x_rho`.
Circuit history: `track_perf` (raw results there), and `track_delta` / `track_qdelta`: how much better than
**his own form at the time** a driver did there (the specialist part, not "had the best car that year").
`team_trend` / `team_qtrend` = fast form (`half_life_fast` races) minus normal form: picks up upgrades.
Chaos: each race gets a 0-1 score for how much less the order stuck than usual at that circuit;
`chaos_form` / `chaos_loss` down-weight chaotic races in the form features / the training loss (both off by default).
(Needs Q1-Q3 times in the data: re-run `python -m core.data` once; it rebuilds from the cache, no download.)

`core/plackett_luce.py` gives each driver a linear score `w · x`. Training maximises the probability
of the observed top-`top_k` order: P(winner | field) × P(2nd | rest) × ...,
where each factor is a softmax. `top_k=1` is plain cross-entropy on the winner.

`--set model="ridge"` swaps the loss for ridge regression (pointwise squared error on each driver's
finishing position, closed-form solution, `alpha` = L2 strength), on exactly the same features.
That isolates "listwise vs pointwise loss" from everything else.

## Writing a predictor

```python
def my_model(hist: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
    # hist: all rows before this race (all columns). race: one row per driver, allowed columns only.
    return scores  # higher = better, aligned with race rows
```

Add it to `PREDICTORS` in the mode's script, or make a new script in that folder.

## Known gaps / TODO

- **This is a first model that only uses previous race and qualifying results.** It does not consider
  weather, tyre strategy, practice / long-run pace, car upgrades, safety-car likelihood, or many other
  things that decide real races.

- Pre-weekend mode knows the actual starters (slightly optimistic vs. real entry list)
