"""From "most likely zone" to "most valuable zone".

The dissertation's central finding is that the classifier is *descriptive*: it
recommends the zone a batter is most likely to hit, which for a good ball is a
defensive one. Its highest-priority future work (Section 7.2) is an
expected-value recommender:

    recommended zone = argmax_z  P(zone = z | x) * V(z | phase, pace/spin)

The difficulty is evaluation, not implementation. If the same zone values are
used to *choose* the recommendation and to *score* it, the evaluation is
circular. This module avoids that with sample splitting:

* Matches are randomly split into two halves, A and B.
* For a delivery in a half-A match, the zone values used to pick the
  recommendation come only from half B.
* The reward model used to score it comes only from half A, excluding the
  delivery's own match.

So the data that chooses the recommendation never judges it, and neither sees
the delivery being evaluated. The argmax policy is evaluated under the same
split so the two are compared like for like.

This extension is not in the dissertation. Its output on the synthetic sample
demonstrates the code only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, ope


def split_matches(matches, seed: int = config.SEED) -> dict:
    """Assign each match to half 'A' or 'B' at random."""
    rng = np.random.default_rng(seed)
    matches = np.array(sorted(pd.unique(matches)))
    perm = rng.permutation(len(matches))
    half = len(matches) // 2
    return {m: ("A" if i < half else "B") for m, i in zip(matches[perm], range(len(matches)))}


def expected_value_zones(df: pd.DataFrame, proba: np.ndarray, value_source: pd.DataFrame,
                         k: int) -> np.ndarray:
    """argmax over zones of P(zone | x) * V(zone | phase, pace), V from ``value_source``."""
    table = ope.build_value_table(value_source)
    values = np.array([[ope.value_lookup(z, ph, bp, table) for z in range(k)]
                       for ph, bp in zip(df["phase"], df["bowler_pace"])])
    return (proba * values).argmax(axis=1)


def compare_policies(deliveries: pd.DataFrame, proba: np.ndarray, k: int = config.K_ZONES,
                     seed: int = config.SEED, n_boot: int = config.N_BOOTSTRAP,
                     by: str | None = "phase", order: list | None = None) -> pd.DataFrame:
    """DR value of the argmax and expected-value policies under sample splitting."""
    df = deliveries.reset_index(drop=True).copy()
    half = df["Match"].map(split_matches(df["Match"], seed))
    df["half"] = half
    df["ev_zone"] = -1
    for h, other in [("A", "B"), ("B", "A")]:
        mask = (half == h).to_numpy()
        df.loc[mask, "ev_zone"] = expected_value_zones(df[mask], proba[mask], df[~mask], k)

    results = []
    for policy, rec_col in [("Most likely zone (dissertation)", "predicted_zone"),
                            ("Expected-value zone (extension)", "ev_zone")]:
        scored = []
        for h in ["A", "B"]:
            mask = (half == h).to_numpy()
            part = ope.add_propensities(df[mask], proba[mask], rec_col=rec_col)
            part = ope.cross_fitted_rewards(part, rec_col=rec_col, train_pool=part)
            scored.append(part)
        scored = pd.concat(scored)
        tbl = ope.evaluate_policy(scored, by=by, order=order, n_boot=n_boot)
        tbl.insert(0, "policy", policy)
        tbl["share_matching_play"] = [scored["follows"].mean()] + (
            [scored.loc[scored[by] == g, "follows"].mean() for g in tbl["group"][1:]] if by else [])
        results.append(tbl)
    return pd.concat(results, ignore_index=True)
