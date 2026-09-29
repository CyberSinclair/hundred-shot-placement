import numpy as np
import pandas as pd
import pytest

from shotzones import cleaning, config, features, repair, synthetic


@pytest.fixture(scope="module")
def clean():
    raw, _ = repair.repair_column_shift(synthetic.generate(n_matches=6, seed=5))
    df, log = cleaning.clean_deliveries(raw)
    return df, log


def test_cleaning_removes_illegal_and_not_hit(clean):
    df, log = clean
    assert df["Legal Ball"].eq("Yes").all()
    assert not df["Connection"].isin(config.NOT_HIT_CONNECTIONS).any()
    assert df[["FieldX", "FieldY"]].notna().all().all()
    lf = log.to_frame()
    assert lf["rows_remaining"].is_monotonic_decreasing
    assert lf["removed"].iloc[1:].sum() == lf["rows_remaining"].iloc[0] - lf["rows_remaining"].iloc[-1]


def test_phase_keeps_the_hundredth_ball():
    br = pd.Series([99, 51, 50, 21, 20, 1, 0])
    ph = cleaning.assign_phase(br)
    assert ph.notna().all()
    assert ph.iloc[-1] == "Death (balls 80-100)"
    assert ph.iloc[2] == "Middle (balls 50-79)"
    assert ph.iloc[1] == "Opening (balls 1-49)"


def test_feature_matrix_has_no_outcome_columns(clean):
    df, _ = clean
    X = features.FeatureBuilder().fit_transform(df)
    features.assert_no_leakage(X.columns)
    assert "Extra Runs" not in X.columns
    assert not any(c.startswith("Feet_") for c in X.columns)
    assert X.isna().sum().sum() == 0


def test_leakage_guard_raises():
    with pytest.raises(ValueError):
        features.assert_no_leakage(["PitchX", "Extra Runs"])
    with pytest.raises(ValueError):
        features.assert_no_leakage(["Shot_Drive"])


def test_imputation_uses_training_rows_only(clean):
    df, _ = clean
    train = df.iloc[: len(df) // 2].copy()
    test = df.iloc[len(df) // 2:].copy()
    test["ImpactZ"] = np.nan
    fb = features.FeatureBuilder().fit(train)
    X = fb.transform(test)
    assert np.allclose(X["ImpactZ"], train["ImpactZ"].median())
    assert list(X.columns) == fb.columns_  # same columns whatever categories test has


def _innings(post_ball: bool):
    runs = [0, 4, 1, 0, 6, 2]
    cum, total = [], 0
    for r in runs:
        if post_ball:
            total += r
            cum.append(total)
        else:
            cum.append(total)
            total += r
    return pd.DataFrame({"Match": "m", "Innings": 1, "Batter ID": "b",
                         "Innings Ball": range(1, 7), "Runs": runs,
                         "Cumulative Batter Runs": cum})


def test_cumulative_timing_check_detects_both_cases():
    assert features.check_cumulative_timing(_innings(False))["verdict"].startswith("pre-ball")
    assert features.check_cumulative_timing(_innings(True))["verdict"].startswith("post-ball")


def test_to_pre_delivery_lags_a_post_ball_total():
    post = _innings(True)
    pre = _innings(False)
    fixed = features.to_pre_delivery(post, "Cumulative Batter Runs")
    assert fixed.tolist() == pre["Cumulative Batter Runs"].tolist()
