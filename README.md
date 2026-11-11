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

## Setup (macOS, uv)

```bash
uv sync                          # creates .venv, installs everything
uv run python -m core.data       # pull 2018-2026 results -> data/results.parquet (first run ~15-30 min, then cached)
uv run python -m core.data --refresh   # after each new race weekend
uv run pytest
```

## Evaluate

```bash
uv run python pre-weekend/baselines.py
uv run python post-quali/baselines.py
uv run python post-quali/baselines.py --seasons 2023 2024 2025 2026
```

**Walk-forward:** every race in `EVAL_SEASONS` is predicted using only races strictly before it.
The evaluator passes the predictor only the columns allowed for its mode
(`PRE_WEEKEND_COLS` / `POST_QUALI_COLS` in `core/config.py`), so it can't see results it shouldn't.

**Holdout:** 2026 US, Mexico City, São Paulo GPs are excluded from tuning. Score them once at the end:
`... baselines.py --holdout`. Then the real test is Las Vegas, Qatar, Abu Dhabi.

## Metrics

- `winner_mrr` – 1 / rank of the actual winner in our order (main winner metric)
- `podium_p@3` – share of the real top 3 we named in our top 3 (main podium metric)
- `podium_set` / `podium_exact` – all three right (any order / exact order)
- `ndcg@3` – order-aware podium score (P1 weighted 3, P2 2, P3 1)

## Writing a predictor

```python
def my_model(hist: pd.DataFrame, race: pd.DataFrame) -> np.ndarray:
    # hist: all rows before this race (all columns). race: one row per driver, allowed columns only.
    return scores  # higher = better, aligned with race rows
```

Add it to `PREDICTORS` in the mode's script, or make a new script in that folder.

## Known gaps / TODO

- Sprint results not pulled yet (championship baseline uses GP points only)
- Pre-weekend mode knows the actual starters (slightly optimistic vs. real entry list)
