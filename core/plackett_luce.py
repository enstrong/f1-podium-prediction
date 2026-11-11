"""Linear scorer trained with a Plackett-Luce (top-k) ranking loss.

score_i = w . x_i                       one number per driver, higher = better

P(this podium) = P(1st | everyone) * P(2nd | everyone but 1st) * P(3rd | everyone but 1st, 2nd)
each factor is a softmax over the drivers still "left":

    P(i next | set S) = exp(score_i) / sum_{j in S} exp(score_j)

loss = -log P(top_k in the right order), averaged over training races (+ L2 on w).
top_k = 1 is plain softmax cross-entropy on "who won" (like a recommender's softmax over
candidates, except here every driver in the field is a candidate - no sampling needed).

No bias term: adding the same constant to every driver changes nothing in a softmax.
"""
import numpy as np
import pandas as pd
import torch

DEFAULT_MODEL_PARAMS = {
    "top_k": 3,             # how many finishing places the loss explains
    "l2": 0.01,             # weight decay on w; sweep on a log scale
    "lr": 0.05,
    "epochs": 300,          # full-batch Adam steps (tiny data: every step sees every race)
    "train_seasons": None,  # only train on the last N seasons (None = all history)
    "recency_half_life": None,  # in seasons: older races get less weight in the loss (None = equal)
    "chaos_loss": 0.0,      # 0..1: down-weight chaotic races in the loss (1 = the most chaotic count ~0)
}


def pack(df: pd.DataFrame, feats: list) -> tuple:
    """One padded tensor per race: X [races, max_drivers, n_feats], rank [races, max_drivers]
    (0 = winner, padding = large), mask [races, max_drivers]."""
    keys = df["season"] * 100 + df["round"]
    groups = [g for _, g in df.assign(_k=keys).groupby("_k", sort=True)]
    R, D, F = len(groups), max(len(g) for g in groups), len(feats)
    X = np.zeros((R, D, F), np.float32)
    rank = np.full((R, D), 10_000, np.int64)
    mask = np.zeros((R, D), bool)
    for r, g in enumerate(groups):
        n = len(g)
        X[r, :n] = g[feats].to_numpy(np.float32)
        rank[r, :n] = g["finish"].to_numpy() - 1
        mask[r, :n] = True
    seasons = np.array([g["season"].iloc[0] for g in groups])
    chaos = np.array([g["race_chaos"].iloc[0] if "race_chaos" in g else 0.0 for g in groups])
    return torch.from_numpy(X), torch.from_numpy(rank), torch.from_numpy(mask), seasons, chaos


def pl_loss(scores: torch.Tensor, rank: torch.Tensor, mask: torch.Tensor, top_k: int,
            race_w: torch.Tensor | None = None) -> torch.Tensor:
    """Negative log-likelihood of the observed top_k order, per race, then (weighted) mean."""
    nll = torch.zeros(scores.shape[0])
    for k in range(top_k):
        left = mask & (rank >= k)                       # drivers not yet placed
        lse = torch.logsumexp(scores.masked_fill(~left, -1e9), dim=1)
        chosen = (scores * (rank == k)).sum(dim=1)      # score of whoever actually finished k-th
        nll = nll + lse - chosen
    if race_w is None:
        return nll.mean()
    return (nll * race_w).sum() / race_w.sum()


class PlackettLuce:
    def __init__(self, feats: list, **params):
        self.feats = feats
        self.p = DEFAULT_MODEL_PARAMS | params
        self.w = None
        self.history = []  # training loss per epoch, for plotting

    def fit(self, train: pd.DataFrame) -> "PlackettLuce":
        p = self.p
        if p["train_seasons"]:
            train = train[train["season"] > train["season"].max() - p["train_seasons"]]
        # standardize so one learning rate suits every feature (and so |w| is comparable)
        self.mu = train[self.feats].mean().to_numpy(np.float32)
        self.sd = train[self.feats].std().replace(0, 1).to_numpy(np.float32)
        X, rank, mask, seasons, chaos = pack(train, self.feats)
        X = (X - torch.from_numpy(self.mu)) / torch.from_numpy(self.sd)

        race_w = np.ones(len(seasons))
        if p["recency_half_life"]:
            race_w *= 0.5 ** ((seasons.max() - seasons) / p["recency_half_life"])
        if p["chaos_loss"]:
            race_w *= 1 - p["chaos_loss"] * chaos
        race_w = torch.from_numpy(race_w).float()

        torch.manual_seed(0)
        w = torch.zeros(len(self.feats), requires_grad=True)
        opt = torch.optim.Adam([w], lr=p["lr"])
        self.history = []
        for _ in range(p["epochs"]):
            opt.zero_grad()
            loss = pl_loss(X @ w, rank, mask, p["top_k"], race_w) + p["l2"] * (w ** 2).sum()
            loss.backward()
            opt.step()
            self.history.append(loss.item())
        self.w = w.detach().numpy()
        return self

    def score(self, race: pd.DataFrame) -> np.ndarray:
        x = (race[self.feats].to_numpy(np.float32) - self.mu) / self.sd
        return x @ self.w

    def weights(self) -> pd.Series:
        """Learned weight per (standardized) feature: sign and size are directly readable."""
        return pd.Series(self.w, index=self.feats).sort_values(key=abs, ascending=False)
