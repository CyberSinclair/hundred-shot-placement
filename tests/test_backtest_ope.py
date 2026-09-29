import numpy as np
import pandas as pd

from shotzones import backtest, ope


def _bandit(n_matches=40, per_match=60, k=4, seed=0):
    """Contextual bandit with a known answer.

    Behaviour picks zones uniformly at random, so mu = 1/k exactly, and zone z
    always scores z runs. The target policy always recommends zone k-1, so its
    true value is k-1 and the behaviour value is (k-1)/2.
    """
    rng = np.random.default_rng(seed)
    n = n_matches * per_match
    zone = rng.integers(0, k, n)
    df = pd.DataFrame({
        "Match": np.repeat([f"m{i}" for i in range(n_matches)], per_match),
        "shot_zone": zone,
        "predicted_zone": k - 1,
        "Runs": zone.astype(float),
        "phase": "Any",
        "bowler_pace": 1,
    })
    proba = np.full((n, k), 1 / k)
    return df, proba


def test_estimators_recover_known_policy_value():
    df, proba = _bandit()
    scored = ope.cross_fitted_rewards(ope.add_propensities(df, proba))
    est = ope.estimates(scored)
    assert abs(est["behaviour"] - 1.5) < 0.1
    for name in ["IPS", "SNIPS", "DR"]:
        assert abs(est[name] - 3.0) < 0.25, name
    # DM uses a shrunk mean, so it is pulled a little toward the global mean.
    assert 2.5 < est["DM"] <= 3.0


def test_bootstrap_interval_brackets_the_estimate():
    df, proba = _bandit()
    scored = ope.cross_fitted_rewards(ope.add_propensities(df, proba))
    lo, hi = ope.cluster_bootstrap_uplift(scored, n_boot=500)
    uplift = ope.estimates(scored)["DR_uplift"]
    assert lo < uplift < hi
    assert lo > 0  # the target policy really is better here


def test_reward_model_never_sees_its_own_match():
    df, proba = _bandit(n_matches=3, per_match=10)
    df.loc[df["Match"] == "m0", "Runs"] = 100.0  # poison one match
    scored = ope.cross_fitted_rewards(ope.add_propensities(df, proba))
    assert scored.loc[scored["Match"] == "m0", "v_rec"].max() < 10


def test_permutation_test_is_null_when_label_is_random():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"Match": rng.integers(0, 30, n), "followed": rng.integers(0, 2, n),
                       "Runs": rng.choice([0, 1, 2, 4, 6], n)})
    res = backtest.permutation_chi_square(df, n_perm=199)
    assert res["p"] > 0.01


def test_cmh_detects_a_planted_effect():
    rng = np.random.default_rng(2)
    n = 4000
    followed = rng.integers(0, 2, n)
    boundary = (rng.random(n) < np.where(followed == 1, 0.30, 0.15)).astype(int)
    df = pd.DataFrame({"Match": rng.integers(0, 40, n), "followed": followed,
                       "is_boundary": boundary})
    r = backtest.cmh(df)
    assert r["odds_ratio"] > 1.8 and r["p"] < 1e-6
