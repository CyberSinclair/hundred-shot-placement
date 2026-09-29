"""Off-policy evaluation of a recommendation policy (dissertation Section 4.8).

Each delivery is one round of a contextual bandit:

* context ``x``: the pre-delivery state (the classifier's features)
* action ``a``: the zone the batter actually played into
* reward ``r``: runs scored

The behaviour policy ``mu(a | x)`` is estimated by the classifier's out-of-fold
probabilities. The target policy ``pi`` is deterministic: it always picks one
recommended zone. Four estimators of the target policy's value are computed:

=========  ==================================================================
DM         mean of the reward model at the recommended zone
IPS        followed deliveries reweighted by 1 / mu(a | x)
SNIPS      IPS divided by the sum of weights rather than n
DR         DM plus an IPS correction applied to the reward model's residual
=========  ==================================================================

Confidence intervals use a cluster bootstrap over matches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_value_table(train: pd.DataFrame, zone_col: str = "shot_zone",
                      alpha_zone: float = config.REWARD_SHRINK_ZONE):
    """Hierarchically shrunk E[runs] by (zone reached, phase, pace/spin).

    Zone means shrink toward the global mean, and each (zone, phase, pace) cell
    shrinks toward its zone mean, so sparse cells borrow strength.
    """
    g = train["Runs"].mean()
    z = train.groupby(zone_col)["Runs"].agg(["mean", "count"])
    zone_v = ((z["mean"] * z["count"] + g * alpha_zone) / (z["count"] + alpha_zone)).to_dict()
    cell = train.groupby([zone_col, "phase", "bowler_pace"], observed=True)["Runs"].agg(["mean", "count"])
    cell_d = {idx: (row["mean"], row["count"]) for idx, row in cell.iterrows()}
    return g, zone_v, cell_d


def value_lookup(zone, phase, pace, table, alpha_cell: float = config.REWARD_SHRINK_CELL) -> float:
    g, zone_v, cell_d = table
    zb = zone_v.get(zone, g)
    mc = cell_d.get((zone, phase, pace))
    if mc is None:
        return zb
    m, n = mc
    return (m * n + zb * alpha_cell) / (n + alpha_cell)


def cross_fitted_rewards(df: pd.DataFrame, rec_col: str = "predicted_zone",
                         train_pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Add ``v_rec`` (value of the recommended zone) and ``v_act`` (value of the
    zone played), each estimated without the delivery's own match.

    ``train_pool`` restricts which rows the reward model may learn from, which
    is how :mod:`shotzones.recommend` keeps recommendation and evaluation on
    separate data.
    """
    pool = df if train_pool is None else train_pool
    out = []
    for m, te in df.groupby("Match", sort=False):
        table = build_value_table(pool[pool["Match"] != m])
        te = te.copy()
        te["v_rec"] = [value_lookup(z, ph, bp, table)
                       for z, ph, bp in zip(te[rec_col], te["phase"], te["bowler_pace"])]
        te["v_act"] = [value_lookup(z, ph, bp, table)
                       for z, ph, bp in zip(te["shot_zone"], te["phase"], te["bowler_pace"])]
        out.append(te)
    return pd.concat(out).loc[df.index]


def add_propensities(df: pd.DataFrame, proba: np.ndarray, rec_col: str = "predicted_zone") -> pd.DataFrame:
    """Behaviour propensity of the action taken and of the recommended action."""
    out = df.copy()
    idx = np.arange(len(out))
    out["p_actual"] = proba[idx, out["shot_zone"].to_numpy()]
    out["p_recommended"] = proba[idx, out[rec_col].to_numpy()]
    out["follows"] = (out["shot_zone"] == out[rec_col]).astype(float)
    return out


def estimates(df: pd.DataFrame, eps: float = config.PROPENSITY_FLOOR) -> pd.Series:
    """Behaviour value and DM / IPS / SNIPS / DR for one set of deliveries.

    Expects columns Runs, follows, p_actual, v_rec, v_act.
    """
    n = len(df)
    p = df["p_actual"].clip(lower=eps)
    f = df["follows"].astype(float)
    r = df["Runs"].astype(float)
    w = f / p
    behaviour = r.mean()
    dr = (df["v_rec"] + w * (r - df["v_act"])).mean()
    return pd.Series({
        "n": n,
        "behaviour": behaviour,
        "DM": df["v_rec"].mean(),
        "IPS": (w * r).sum() / n,
        "SNIPS": (w * r).sum() / w.sum() if w.sum() > 0 else np.nan,
        "DR": dr,
        "DR_uplift": dr - behaviour,
    })


def cluster_bootstrap_uplift(df: pd.DataFrame, n_boot: int = config.N_BOOTSTRAP,
                             seed: int = config.SEED, eps: float = config.PROPENSITY_FLOOR,
                             level: float = 0.95) -> tuple[float, float]:
    """Percentile CI for DR uplift, resampling whole matches with replacement."""
    p = df["p_actual"].clip(lower=eps).to_numpy()
    f = df["follows"].to_numpy(dtype=float)
    r = df["Runs"].to_numpy(dtype=float)
    dr_term = df["v_rec"].to_numpy() + f * (r - df["v_act"].to_numpy()) / p
    codes, _ = pd.factorize(df["Match"])
    M = codes.max() + 1
    n_m = np.bincount(codes, minlength=M).astype(float)
    r_m = np.bincount(codes, weights=r, minlength=M)
    dr_m = np.bincount(codes, weights=dr_term, minlength=M)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, M, size=(n_boot, M))
    uplift = (dr_m[idx].sum(axis=1) - r_m[idx].sum(axis=1)) / n_m[idx].sum(axis=1)
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(uplift, [a, 100 - a])
    return float(lo), float(hi)


def evaluate_policy(df: pd.DataFrame, by: str | None = None, order: list | None = None,
                    n_boot: int = config.N_BOOTSTRAP) -> pd.DataFrame:
    """Estimates with bootstrap CI, pooled and optionally per level of ``by``."""
    groups = [("All deliveries", df)]
    if by:
        levels = order or sorted(df[by].dropna().unique())
        groups += [(lvl, df[df[by] == lvl]) for lvl in levels]
    rows = []
    for label, sub in groups:
        if sub.empty:
            continue
        s = estimates(sub)
        lo, hi = cluster_bootstrap_uplift(sub, n_boot=n_boot)
        rows.append({"group": label, **s.to_dict(), "uplift_ci_low": lo, "uplift_ci_high": hi})
    return pd.DataFrame(rows)
