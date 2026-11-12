# 2026 R21 Las Vegas Grand Prix: pre-weekend (before FP1)

- Generated: 2026-11-12 08:02 UTC, race date 2026-11-22
- Data through: 2026 R20 Brazilian Grand Prix
- Code: 84bb495
- Scope: model 2, built only from previous race and qualifying results. It does NOT use weather, tyre strategy, practice pace, upgrades or many other things that decide races.
- Model: model 2, LightGBM LambdaRank (P(win) = softmax(score / T), T = 0.798 fitted on held-out races), params {'half_life': 6.0, 'half_life_fast': 2.0, 'season_carry': 0.7, 'reg_carry': 0.25, 'shrink': 1.0, 'track_shrink': 2.0, 'circuit_shrink': 10.0, 'chaos_form': 0.0, 'n_trees': 100, 'lgbm_lr': 0.05, 'num_leaves': 7, 'min_child_samples': 20, 'subsample': 0.8, 'colsample': 0.8, 'lgbm_l2': 1.0, 'labels': 'podium', 'train_seasons': None, 'drop': [], 'model': 'lgbm'}

**Podium pick: VER / NOR / RUS**

| # | Driver | Team | P(win) | P(podium) |
|---|---|---|---|---|
| 1 | VER | Red Bull | 31.2% | 75.0% |
| 2 | NOR | McLaren | 19.8% | 59.2% |
| 3 | RUS | Mercedes | 15.9% | 50.4% |
| 4 | ANT | Mercedes | 14.6% | 47.2% |
| 5 | HAD | Red Bull | 5.5% | 20.3% |
| 6 | LEC | Ferrari | 4.4% | 16.2% |
| 7 | HAM | Ferrari | 3.4% | 12.7% |
| 8 | PIA | McLaren | 3.4% | 12.3% |
| 9 | LAW | RB F1 Team | 0.4% | 1.7% |
| 10 | LIN | RB F1 Team | 0.2% | 1.0% |
| 11 | GAS | Alpine F1 Team | 0.2% | 0.8% |
| 12 | COL | Alpine F1 Team | 0.1% | 0.3% |
| 13 | CAM | Haas F1 Team | 0.1% | 0.3% |
| 14 | HUL | Audi | 0.1% | 0.3% |
| 15 | BOR | Audi | 0.1% | 0.2% |
| 16 | BOT | Cadillac F1 Team | 0.1% | 0.3% |
| 17 | ALB | Williams | 0.1% | 0.3% |
| 18 | STR | Aston Martin | 0.1% | 0.3% |
| 19 | PER | Cadillac F1 Team | 0.1% | 0.2% |
| 20 | ALO | Aston Martin | 0.1% | 0.3% |
| 21 | BEA | Haas F1 Team | 0.1% | 0.2% |
| 22 | SAI | Williams | 0.1% | 0.2% |
