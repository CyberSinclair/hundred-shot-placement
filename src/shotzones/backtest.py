"""Counterfactual backtesting of the recommendation (dissertation Section 4.6).

A delivery "followed" the recommendation when the zone the batter actually hit
into equals the model's predicted zone. Run outcomes are then compared between
followed and not-followed deliveries.

Three tests, arranged so the effect of within-match clustering can be seen:

1. Naive chi-square on the 2 x 6 table (treats deliveries as independent).
2. Permutation chi-square, shuffling the followed label *within* each match.
3. Cochran-Mantel-Haenszel stratified by match, on a binary outcome, with the
   Breslow-Day test of homogeneity.

Caution on per-zone breakdowns: within a recommended zone, "followed" means the
ball *reached* that zone. For boundary-area zones that almost guarantees more
boundaries, and for central zones it almost rules them out. The per-zone odds
ratios therefore mostly restate the zone definitions and should not be read as
evidence that a recommendation worked.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

from . import config


def followed_summary(df: pd.DataFrame, followed_col: str = "followed") -> pd.DataFrame:
    f = df[followed_col].eq(1)
    return pd.DataFrame({
        "group": ["Followed recommendation", "Did not follow"],
        "n": [int(f.sum()), int((~f).sum())],
        "mean_runs": [df.loc[f, "Runs"].mean(), df.loc[~f, "Runs"].mean()],
        "boundary_pct": [df.loc[f, "is_boundary"].mean() * 100,
                         df.loc[~f, "is_boundary"].mean() * 100],
    })


def run_distribution(df: pd.DataFrame, followed_col: str = "followed") -> pd.DataFrame:
    """Row percentages of each run value by group (Table 5.10)."""
    ct = pd.crosstab(df[followed_col], df["Runs"]).reindex(
        index=[0, 1], columns=config.RUN_VALUES, fill_value=0)
    ct.index = ["Not followed", "Followed"]
    return ct.div(ct.sum(axis=1), axis=0) * 100


def _chi2(followed: np.ndarray, runs: np.ndarray) -> float:
    tab = pd.crosstab(followed, runs).reindex(index=[0, 1], columns=config.RUN_VALUES,
                                              fill_value=0).to_numpy()
    tab = tab[:, tab.sum(axis=0) > 0]
    return stats.chi2_contingency(tab)[0]


def naive_chi_square(df: pd.DataFrame, followed_col: str = "followed") -> dict:
    tab = pd.crosstab(df[followed_col], df["Runs"]).reindex(
        index=[0, 1], columns=config.RUN_VALUES, fill_value=0).to_numpy()
    tab = tab[:, tab.sum(axis=0) > 0]
    chi2, p, dof, expected = stats.chi2_contingency(tab)
    v = np.sqrt(chi2 / (tab.sum() * (min(tab.shape) - 1)))
    return {"statistic": chi2, "p": p, "dof": dof, "cramers_v": v,
            "min_expected": float(expected.min())}


def permutation_chi_square(df: pd.DataFrame, n_perm: int = config.N_PERMUTATIONS,
                           seed: int = config.SEED, followed_col: str = "followed") -> dict:
    """Shuffle the followed label within each match, keeping every match's own
    scoring profile and follow rate fixed."""
    rng = np.random.default_rng(seed)
    followed = df[followed_col].to_numpy()
    runs = df["Runs"].to_numpy()
    groups = [idx for idx in df.groupby("Match").indices.values()]
    observed = _chi2(followed, runs)
    null = np.empty(n_perm)
    for i in range(n_perm):
        sh = followed.copy()
        for idx in groups:
            sh[idx] = rng.permutation(followed[idx])
        null[i] = _chi2(sh, runs)
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return {"statistic": observed, "p": p, "null_95th": float(np.percentile(null, 95)),
            "n_perm": n_perm}


def cmh(df: pd.DataFrame, outcome_col: str = "is_boundary",
        followed_col: str = "followed", strata_col: str = "Match") -> dict | None:
    """CMH test stratified by match with pooled odds ratio and Breslow-Day.

    Strata with an empty row or column carry no information and are skipped.
    Returns None if fewer than two informative strata remain.
    """
    tables = []
    for _, g in df.groupby(strata_col):
        t = pd.crosstab(g[followed_col], g[outcome_col]).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0).to_numpy()
        if t.sum(axis=1).min() > 0 and t.sum(axis=0).min() > 0:
            tables.append(t.astype(float))
    if len(tables) < 2:
        return None
    # Zones that can never produce a boundary give degenerate tables; statsmodels
    # warns and returns nan/0/inf, which is the honest answer, so silence it.
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        st = StratifiedTable(tables)
        test = st.test_null_odds()
        lo, hi = st.oddsratio_pooled_confint()
        try:
            bd_p = st.test_equal_odds().pvalue
        except Exception:  # Breslow-Day can fail on degenerate strata
            bd_p = np.nan
    return {"statistic": test.statistic, "p": test.pvalue, "odds_ratio": st.oddsratio_pooled,
            "ci_low": lo, "ci_high": hi, "breslow_day_p": bd_p, "strata": len(tables)}


def hypothesis_tests(df: pd.DataFrame, n_perm: int = config.N_PERMUTATIONS) -> pd.DataFrame:
    """Table 5.11: all three tests on the pooled data."""
    naive = naive_chi_square(df)
    perm = permutation_chi_square(df, n_perm=n_perm)
    rows = [
        {"test": "Chi-square (naive, 2 x 6)", "statistic": naive["statistic"],
         "p": naive["p"], "effect": f"Cramer's V = {naive['cramers_v']:.4f}"},
        {"test": "Permutation chi-square (within match)", "statistic": perm["statistic"],
         "p": perm["p"], "effect": ""},
    ]
    for label, col in [("CMH, boundary vs not", "is_boundary"),
                       ("CMH, scoring shot vs dot", "is_scoring")]:
        r = cmh(df, col)
        if r:
            rows.append({"test": label, "statistic": r["statistic"], "p": r["p"],
                         "effect": f"OR = {r['odds_ratio']:.3f} "
                                   f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]",
                         "breslow_day_p": r["breslow_day_p"]})
    return pd.DataFrame(rows)


def breakdown(df: pd.DataFrame, by: str, order: list | None = None,
              outcome_col: str = "is_boundary") -> pd.DataFrame:
    """Followed vs not within each level of ``by`` (Tables 5.12 to 5.14)."""
    levels = order or sorted(df[by].dropna().unique())
    rows = []
    for level in levels:
        sub = df[df[by] == level]
        if sub.empty:
            continue
        f = sub["followed"].eq(1)
        r = cmh(sub, outcome_col) or {}
        rows.append({
            by: level, "n": len(sub),
            "runs_followed": sub.loc[f, "Runs"].mean(),
            "runs_not_followed": sub.loc[~f, "Runs"].mean(),
            "bnd_pct_followed": sub.loc[f, "is_boundary"].mean() * 100,
            "bnd_pct_not_followed": sub.loc[~f, "is_boundary"].mean() * 100,
            "odds_ratio": r.get("odds_ratio", np.nan),
            "ci_low": r.get("ci_low", np.nan), "ci_high": r.get("ci_high", np.nan),
            "p": r.get("p", np.nan), "strata": r.get("strata", 0),
        })
    return pd.DataFrame(rows)
