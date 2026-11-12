"""Model 2: LightGBM with the LambdaRank objective, on the same features as model 1.

Model 1 (Plackett-Luce) is a straight line: score = w . x, 1 weight per feature.
Model 2 is a SUM OF SMALL DECISION TREES:

    score_i = eta * (tree_1(x_i) + tree_2(x_i) + ... + tree_T(x_i))

Each tree asks a few yes/no questions ("grid_perf > 0.85?", "team_trend > 0.03?") and gives a
number for each leaf. So it can learn interactions nobody hand-built, e.g. "fast car AND starting
at the back AND high-overtaking circuit", and thresholds ("only the front row matters at Monaco").

How the trees are trained (boosting): tree t is fitted to "which way and how much should each
driver's current score move", i.e. the gradient of the loss. LambdaRank defines that gradient:
for every pair of drivers in a race where i finished ahead of j,

    lambda_ij = -sigma / (1 + exp(sigma * (s_i - s_j)))  *  |delta NDCG_ij|

The first part is RankNet: large when the model has j ahead of i (wrong), ~0 when i is already
comfortably ahead. |delta NDCG_ij| = how much our NDCG@k metric would change if i and j swapped
places in the current ranking: big for pairs that touch the podium, ~0 for P14 vs P15.
So the trees spend their effort on getting the top of the order right.

Scores are NOT probabilities. For P(win) we fit one temperature T on recent races that were not
used for training: P(win) = softmax(score / T). See calibrate().
"""
import numpy as np
import pandas as pd

DEFAULT_LGBM_PARAMS = {
    "n_trees": 100,             # T: how many trees are added up (sweep: 200 overfits our ~4k rows)
    "lgbm_lr": 0.05,            # eta: how much of each tree's suggestion we take (small = cautious)
    "num_leaves": 7,            # leaves per tree: 7 leaves ~ 2-3 questions deep
    "min_child_samples": 20,    # a leaf must contain >= 20 driver-race rows (stops memorising single races)
    "subsample": 0.8,           # each tree sees a random 80% of races' rows ...
    "colsample": 0.8,           # ... and a random 80% of features: decorrelates the trees
    "lgbm_l2": 1.0,             # L2 penalty on leaf values (same idea as ridge's alpha)
    "labels": "podium",         # "podium": P1=3, P2=2, P3=1, else 0  |  "top6": P1=5 .. P6=1
    "train_seasons": None,
}

LABELS = {
    "podium": lambda f: np.clip(4 - f, 0, None),
    "top6": lambda f: np.clip(6 - f, 0, None),
}


class LambdaRank:
    def __init__(self, feats: list, **params):
        self.feats = feats
        self.p = DEFAULT_LGBM_PARAMS | {k: v for k, v in params.items() if k in DEFAULT_LGBM_PARAMS}
        self.history = []

    def _lgb_params(self) -> dict:
        p = self.p
        k = 3 if p["labels"] == "podium" else 6
        return {
            "objective": "lambdarank", "learning_rate": p["lgbm_lr"], "num_leaves": p["num_leaves"],
            "min_data_in_leaf": p["min_child_samples"], "bagging_fraction": p["subsample"], "bagging_freq": 1,
            "feature_fraction": p["colsample"], "lambda_l2": p["lgbm_l2"],
            "lambdarank_truncation_level": k + 2,      # only pairs that involve the top ~k matter
            "label_gain": [0, 1, 3, 7, 15, 31, 63],    # gain of label y = 2^y - 1 (the NDCG convention)
            "seed": 0, "deterministic": True, "force_row_wise": True, "num_threads": 1, "verbose": -1,
        }

    @staticmethod
    def _grouped(df: pd.DataFrame) -> tuple:
        df = df.sort_values(["season", "round"], kind="stable")
        groups = df.groupby(["season", "round"], sort=False).size().to_numpy()   # rows per race
        return df, groups

    def fit(self, train: pd.DataFrame) -> "LambdaRank":
        if self.p["train_seasons"]:
            train = train[train["season"] > train["season"].max() - self.p["train_seasons"]]
        train, groups = self._grouped(train)
        import lightgbm as lgb
        y = LABELS[self.p["labels"]](train["finish"].to_numpy()).astype(int)
        ds = lgb.Dataset(train[self.feats].to_numpy(float), y, group=groups, feature_name=self.feats)
        self.m = lgb.train(self._lgb_params(), ds, num_boost_round=self.p["n_trees"])
        return self

    def score(self, race: pd.DataFrame) -> np.ndarray:
        return self.m.predict(race[self.feats].to_numpy(float))

    def weights(self) -> pd.Series:
        """Total loss reduction from splits on each feature ("gain" importance). Unlike PL weights,
        there's no sign: trees can use a feature in either direction, or only above a threshold."""
        imp = self.m.feature_importance(importance_type="gain")
        return (pd.Series(imp / max(imp.sum(), 1e-12), index=self.feats)).sort_values(ascending=False)

    def calibrate(self, train: pd.DataFrame, last_n: int = 30) -> float:
        """Temperature T for P(win) = softmax(score / T). Train a copy WITHOUT the last `last_n`
        races, score those races, and pick the T that maximises the log-probability given to the
        actual winners (a 1-parameter fit). In-sample scores would be overconfident."""
        keys = (train["season"] * 100 + train["round"])
        cut = np.sort(keys.unique())[-last_n]
        early = LambdaRank(self.feats, **self.p).fit(train[keys < cut])
        late = train[keys >= cut]
        races = [(early.score(g), g["finish"].to_numpy() == 1) for _, g in late.groupby(keys[keys >= cut])]
        grid = np.exp(np.linspace(np.log(0.05), np.log(20), 200))

        def nll(T):
            return -sum((s / T)[w].sum() - np.log(np.exp((s - s.max()) / T).sum()) - s.max() / T for s, w in races)

        self.T = float(grid[np.argmin([nll(T) for T in grid])])
        return self.T
