"""Causal pre-weekend features: every value for race t uses only races strictly before t.

The core trick is one weight matrix W[t, s] = how much race s counts when describing race t:

    W[t, s] = 0.5 ** (races_between / half_life)      # recent races matter more
            * season_carry ** seasons_crossed          # a new season = partial reset
            * reg_carry    ** reg_changes_crossed      # a new rulebook = much bigger reset
            (and 0 when s >= t, which is what makes it causal)

A "form" feature is then a weighted average:  (W @ values) / (W @ played), shrunk towards
a neutral prior when there's little evidence (rookies, start of a regulation era):

    form = (W @ values + shrink * prior) / (W @ played + shrink)

This replaces a hand-made "last season's standing" feature: last season is just older,
discounted history, and the discount is much stronger when the regs changed.
"""
import numpy as np
import pandas as pd

from .config import REG_CHANGE_SEASONS

DEFAULT_FEATURE_PARAMS = {
    "half_life": 6.0,      # races; how fast form fades within an era
    "half_life_fast": 2.0,  # races; a second, faster team form - (fast - slow) = upgrade trend
    "season_carry": 0.7,   # multiplier per season boundary crossed
    "reg_carry": 0.25,     # extra multiplier per regulation reset crossed
    "shrink": 1.0,         # pseudo-races of neutral prior added to every average
    "track_shrink": 2.0,   # pseudo-visits of prior added to circuit-history features (~8 visits max)
    "circuit_shrink": 10.0,  # pseudo-races of the all-circuit average added to each circuit's pole-win / rho
    "chaos_form": 0.0,     # 0..1: how much chaotic races are down-weighted inside form features
}

FEATURES = [
    "driver_perf",    # finishing position mapped to 1 (won) .. 0 (last), weighted average
    "driver_podium",  # weighted podium rate
    "driver_win",     # weighted win rate
    "team_perf",      # same three for the team (mean over its cars) - the car matters most
    "team_podium",
    "team_win",
    "driver_dnf",     # reliability / crash rate
    "evidence",       # log(1 + effective races of driver history): lets the model trust form less for rookies
    "driver_quali",   # qualifying POSITION mapped to 1 (pole) .. 0 (last), weighted average.
    "team_quali",     #   positions, not lap times: a wet Q3 can't make a Q1 exit look faster than pole
    "driver_qgap",    # weighted avg of % gap to the fastest time, in the deepest session reached (see quali_gap)
    "team_qgap",      #   -> "how far off", not just "which position": pole by 0.4s != pole by 0.01s
    "team_trend",     # team race form over the last ~2 races minus over ~6: positive = car just got better
    "team_qtrend",    #   same with quali position (cleaner: no crashes / strategy)
    "track_perf",     # driver's past finishing results at THIS circuit (wins here every year -> high)
    "track_delta",    # ... relative to his own form at the time: HAM at Silverstone, LEC at Monaco
    "track_qdelta",   #   (the "specialist" part, not just "was in the best car back then")
    "quali_x_pole",   # driver_quali * circuit_pole_rate: quali form matters more where pole wins
    "quali_x_rho",    # driver_quali * circuit_rho: ... and where the whole order sticks (Monaco, Zandvoort)
]

# Post-quali adds the actual grid. Circuit-level numbers (pole rate, rho) are the same for every
# driver in a race, so a softmax ignores them on their own - they only matter through interactions.
POST_QUALI_FEATURES = FEATURES + [
    "grid_perf",      # grid slot mapped to 1 (pole) .. 0 (last)
    "grid_x_pole",    # grid_perf * circuit_pole_rate: how often the pole-sitter converts here
    "grid_x_rho",     # grid_perf * circuit_rho: how much the WHOLE grid order survives the race here
    "pitlane",        # started from the pit lane
    "qgap",           # THIS race's quali % gap to the fastest (deepest session reached)
]

PRIOR = {"perf": 0.5, "podium": 0.15, "win": 0.05, "dnf": 0.15, "quali": 0.5, "qgap": 1.5}
QGAP_CAP = 5.0  # % - beyond this it's a crash / no proper lap, not pace


def quali_gap(df: pd.DataFrame) -> pd.Series:
    """% gap to the fastest time, measured in the deepest session each driver reached.

    Q3 drivers are compared with Q3 times, Q2 exits with Q2 times, Q1 exits with Q1 times.
    Track evolution makes later sessions faster, and top drivers often cruise through Q1, so
    comparing everyone's Q1 time would be misleading. Each session is internally consistent
    (same track state, mostly same weather), so its gap is a fair "how far off".
    No forced ordering: if P10 botched their Q3 lap and P11's Q2 lap was close to the best
    Q2 time, P11 can show a smaller gap. That is real pace information; the ORDER is already
    in driver_quali / grid.
    Positions are kept separately (driver_quali) and remain the backbone in wet sessions."""
    key = df["season"] * 100 + df["round"]
    q1, q2, q3 = (df.get(c, pd.Series(np.nan, index=df.index)) for c in ("q1", "q2", "q3"))
    sess = np.select([q3.notna(), q2.notna(), q1.notna()], [3, 2, 1], 0)
    t = q3.where(sess == 3, q2.where(sess == 2, q1.where(sess == 1)))
    # fastest lap of the WHOLE session (everyone who ran it), not just of those knocked out in it
    best = pd.Series(np.select([sess == 3, sess == 2, sess == 1],
                               [q.groupby(key).transform("min") for q in (q3, q2, q1)], np.nan), index=df.index)
    gap = 100 * (t / best - 1)
    gap = gap.fillna(gap.groupby(key).transform("max"))   # no lap time set -> as slow as the slowest
    return gap.clip(0, QGAP_CAP)


GLOBAL_POLE_WIN_PRIOR = 0.5  # only used before any race has been seen


def circuit_pole_rate(races: pd.DataFrame, pole_won: np.ndarray, k: float) -> np.ndarray:
    return circuit_mean(races, pole_won, k, GLOBAL_POLE_WIN_PRIOR)


def circuit_mean(races: pd.DataFrame, values: np.ndarray, k: float, fallback: float) -> np.ndarray:
    """For race t: mean of a per-race value over earlier races AT THIS CIRCUIT, shrunk towards
    the mean over ALL earlier races. E.g. for pole_won (1/0): share of earlier races AT THIS CIRCUIT won from pole, shrunk towards the
    share over ALL earlier races. Like a Bayesian average:
        (pole wins here + k * global rate) / (races here + k)
    With ~8 races per circuit, k=10 means the circuit's own history and the global rate
    count about equally - Monza's raw 2/9 becomes ~0.39, Monaco's 6/8 ~0.63."""
    t = np.arange(len(races))
    before = (t[:, None] > t[None, :]).astype(float)
    cid = races["circuit_id"].to_numpy()
    same = before * (cid[:, None] == cid[None, :])
    seen = (~np.isnan(values)).astype(float)
    v = np.nan_to_num(values)
    g_n, g_w = before @ seen, before @ v
    glob = np.where(g_n > 0, g_w / np.maximum(g_n, 1), fallback)
    return (same @ v + k * glob) / (same @ seen + k)


def race_rho(df: pd.DataFrame, n_races: int) -> np.ndarray:
    """Per race: correlation between grid slot and finishing position (both are ranks, so this
    is ~Spearman). 1 = finished exactly as they started (a processional Monaco), ~0 = lottery
    (wet Interlagos 2024). NaN for races with no result yet."""
    d = df.dropna(subset=["grid", "finish"]) if "grid" in df else df.iloc[:0]
    rho = d.groupby("_r")[["grid", "finish"]].apply(lambda g: g["grid"].corr(g["finish"]))
    return rho.reindex(range(n_races)).to_numpy(float)


def chaos_score(rho: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """How much LESS the order stuck than usual at that circuit, as a 0..1 percentile over all
    races seen. A shuffled Monaco is chaos; a shuffled Monza is less surprising."""
    raw = expected - rho
    out = np.full_like(raw, np.nan)
    seen = []
    for i, r in enumerate(raw):           # percentile among races up to and including i: causal
        if np.isnan(r):
            continue
        seen.append(r)
        out[i] = np.mean(np.array(seen) <= r) if len(seen) > 5 else 0.5
    return out


def _weights(races: pd.DataFrame, p: dict) -> np.ndarray:
    """W[t, s] for the races table (one row per race, chronological)."""
    idx = np.arange(len(races))
    season = races["season"].to_numpy()
    gap = idx[:, None] - idx[None, :]                       # races between s and t
    seasons_crossed = season[:, None] - season[None, :]
    reg = _reg_crossed(season)
    w = 0.5 ** (gap / p["half_life"]) * p["season_carry"] ** seasons_crossed * p["reg_carry"] ** reg
    return np.where(gap > 0, w, 0.0)


def _reg_crossed(season: np.ndarray) -> np.ndarray:
    # count of regulation resets r with season_s < r <= season_t, via cumulative counts per season
    eras = np.searchsorted(np.sort(REG_CHANGE_SEASONS), season, side="right")
    return eras[:, None] - eras[None, :]


def _form(W: np.ndarray, vals: pd.DataFrame, prior: float, shrink: float) -> tuple:
    played = vals.notna().to_numpy(float)
    v = vals.fillna(0).to_numpy(float)
    num, den = W @ v, W @ played
    return (num + shrink * prior) / (den + shrink), den


def build(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Features for every row of df. Rows with finish = NaN (the race being predicted) are fine:
    they get features from the past and contribute nothing to anyone's history."""
    p = DEFAULT_FEATURE_PARAMS | (params or {})
    df = df.copy()
    df["key"] = df["season"] * 100 + df["round"]
    races = df[["key", "season", "circuit_id"]].drop_duplicates("key").sort_values("key").reset_index(drop=True)
    R = len(races)
    ri = pd.Series(races.index, index=races["key"])
    df["_r"] = df["key"].map(ri)
    idx = pd.MultiIndex.from_arrays([df["_r"], df["driver_id"]])

    n = df.groupby("key")["driver_id"].transform("size")
    df["perf"] = 1 - (df["finish"] - 1) / (n - 1)
    df["podium"] = (df["finish"] <= 3).astype(float).where(df["finish"].notna())
    df["win"] = (df["finish"] == 1).astype(float).where(df["finish"].notna())
    df["dnf_"] = df["dnf"].astype(float).where(df["finish"].notna()) if "dnf" in df else np.nan
    if "quali" not in df:
        df["quali"] = np.nan
    nq = df.groupby("key")["quali"].transform("count")
    df["quali_"] = 1 - (df["quali"] - 1) / (nq - 1).clip(lower=1)
    df["qgap_now"] = quali_gap(df)
    df["qgap_"] = df["qgap_now"]
    # the race being predicted has no finish yet: its own quali must not feed its own form
    df.loc[df["finish"].isna(), ["quali_", "qgap_"]] = np.nan

    # --- circuit character + chaos (all causal: race t only sees earlier races) ---
    rho = race_rho(df, R)
    circuit_rho = circuit_mean(races, rho, p["circuit_shrink"], fallback=0.5)
    chaos = chaos_score(rho, circuit_rho)                 # judged against what was expected for that track
    keep = 1 - p["chaos_form"] * np.nan_to_num(chaos)      # per past race: how much it counts in form

    W = _weights(races, p) * keep[None, :]
    W_fast = _weights(races, p | {"half_life": p["half_life_fast"]}) * keep[None, :]

    def form(Wm, col, m, prior, shrink):
        g = df.pivot_table(index="_r", columns=col, values=m, aggfunc="mean", dropna=False).reindex(range(R))
        f, den = _form(Wm, g, prior, shrink)
        look = pd.MultiIndex.from_arrays([df["_r"], df[col]])
        return (pd.DataFrame(f, columns=g.columns).stack().reindex(look).to_numpy(),
                pd.DataFrame(den, columns=g.columns).stack().reindex(look).to_numpy())

    out = pd.DataFrame(index=df.index)
    for who, col in [("driver", "driver_id"), ("team", "team_id")]:
        for m in ["perf", "podium", "win", "quali_", "qgap_"] + (["dnf_"] if who == "driver" else []):
            out[f"{who}_{m.rstrip('_')}"], den = form(W, col, m, PRIOR[m.rstrip("_")], p["shrink"])
            if who == "driver" and m == "perf":
                out["evidence"] = np.log1p(den)
    for m in ["perf", "quali_"]:
        fast, _ = form(W_fast, "team_id", m, PRIOR[m.rstrip("_")], p["shrink"])
        out["team_" + ("trend" if m == "perf" else "qtrend")] = fast - out[f"team_{m.rstrip('_')}"]

    # --- circuit history: same circuit, earlier visits, gently discounted per season ---
    t = np.arange(R)
    same = races["circuit_id"].to_numpy()[:, None] == races["circuit_id"].to_numpy()[None, :]
    season = races["season"].to_numpy()
    Wt = (t[:, None] > t[None, :]) * same * p["season_carry"] ** ((season[:, None] - season[None, :]) / 2)
    Wt = Wt * keep[None, :]
    k = p["track_shrink"]
    out["track_perf"], _ = form(Wt, "driver_id", "perf", PRIOR["perf"], k)
    # specialist = how much better than his OWN form at the time he did here
    df["perf_delta"] = df["perf"] - out["driver_perf"]
    df["quali_delta"] = df["quali_"] - out["driver_quali"]
    out["track_delta"], _ = form(Wt, "driver_id", "perf_delta", 0.0, k)
    out["track_qdelta"], _ = form(Wt, "driver_id", "quali_delta", 0.0, k)

    # --- circuit-level interactions ---
    pole = df[df["grid"] == 1].groupby("_r")["finish"].first() if "grid" in df else pd.Series(dtype=float)
    pole_won = (pole == 1).astype(float).where(pole.notna()).reindex(range(R)).to_numpy(float)
    r_idx = df["_r"].to_numpy()
    out["circuit_pole_rate"] = circuit_pole_rate(races, pole_won, p["circuit_shrink"])[r_idx]
    out["circuit_rho"] = circuit_rho[r_idx]
    out["race_chaos"] = np.nan_to_num(chaos[r_idx])       # used to weight the training loss, not a feature
    out["quali_x_pole"] = out["driver_quali"] * out["circuit_pole_rate"]
    out["quali_x_rho"] = out["driver_quali"] * out["circuit_rho"]

    if "grid" in df:
        out["grid_perf"] = 1 - (df["grid"] - 1) / (n - 1)
        out["grid_x_pole"] = out["grid_perf"] * out["circuit_pole_rate"]
        out["grid_x_rho"] = out["grid_perf"] * out["circuit_rho"]
        out["pitlane"] = df.get("pitlane_start", pd.Series(False, index=df.index)).astype(float)
    out["qgap"] = df["qgap_now"].fillna(PRIOR["qgap"])
    return out.fillna({"evidence": 0.0, "track_delta": 0.0, "track_qdelta": 0.0, "team_trend": 0.0,
                       "team_qtrend": 0.0})
