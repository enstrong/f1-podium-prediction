"""Ridge regression: the pointwise alternative to Plackett-Luce, on the same features.

Each driver-race row is ONE independent example: predict a number y for it.
    y = perf = 1 (won) .. 0 (last)          (or points, which only cares about the top 10)

Loss (squared error + L2 penalty, the "ridge"):
    L(w) = sum_rows (y - x.w)^2 + alpha * |w|^2

It has a closed-form answer, no gradient descent needed:
    w = (X^T X + alpha * I)^-1  X^T y

alpha = 0 is ordinary least squares. Bigger alpha shrinks w towards 0, which helps when
features are correlated (driver_perf / team_perf / driver_quali all say "fast car") and
OLS would give them big opposite-signed weights that cancel out.

The difference from Plackett-Luce: ridge spends as much effort getting P14 vs P15 right
as P1 vs P2, and never compares drivers within a race. PL only scores the top_k places and
always compares each driver against the rest of the field in that race.
"""
import numpy as np
import pandas as pd

TARGETS = {
    "perf": lambda df: 1 - (df["finish"] - 1) / (df.groupby(["season", "round"])["finish"].transform("size") - 1),
    "points": lambda df: df["points"],
}


class Ridge:
    def __init__(self, feats: list, alpha: float = 10.0, target: str = "perf", train_seasons=None, **_):
        self.feats, self.alpha, self.target, self.train_seasons = feats, alpha, target, train_seasons

    def fit(self, train: pd.DataFrame) -> "Ridge":
        if self.train_seasons:
            train = train[train["season"] > train["season"].max() - self.train_seasons]
        self.mu = train[self.feats].mean().to_numpy()
        self.sd = train[self.feats].std().replace(0, 1).to_numpy()
        X = (train[self.feats].to_numpy(float) - self.mu) / self.sd
        y = TARGETS[self.target](train).to_numpy(float)
        y = y - y.mean()                       # centre y, so no intercept is needed
        if self.alpha == 0:
            # plain OLS: X^T X is often singular here (early on, some features are constant or
            # perfectly correlated) - that's exactly the problem the + alpha*I fixes. lstsq picks
            # the smallest-norm solution so we can still compare.
            self.w = np.linalg.lstsq(X, y, rcond=None)[0]
        else:
            self.w = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ y)
        self.history = []
        return self

    def score(self, race: pd.DataFrame) -> np.ndarray:
        return ((race[self.feats].to_numpy(float) - self.mu) / self.sd) @ self.w

    def weights(self) -> pd.Series:
        return pd.Series(self.w, index=self.feats).sort_values(key=abs, ascending=False)
