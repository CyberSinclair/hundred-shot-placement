import numpy as np
import pandas as pd

from shotzones import config, repair, synthetic


def test_synthetic_defect_is_detected_on_2023_and_2024_only():
    raw = synthetic.generate(n_matches=6, seed=1)
    mask = repair.detect_shifted_rows(raw)
    years = pd.to_datetime(raw["Date"]).dt.year
    assert mask.sum() > 0
    assert set(years[mask]) == {2023, 2024}
    assert not mask[years == 2025].any()


def test_repair_restores_the_clean_values_exactly():
    clean = synthetic.generate(n_matches=6, seed=2, with_defects=False)
    broken = synthetic.generate(n_matches=6, seed=2, with_defects=True)
    fixed, report = repair.repair_column_shift(broken)

    assert report.rows_repaired == int((pd.to_datetime(clean["Date"]).dt.year < 2025).sum())
    assert report.sentinel_remaining == 0
    assert report.ball_type_restored == 1.0

    # Every column that survived the truncation must match the defect-free export.
    start = list(clean.columns).index(config.SHIFT_START_COLUMN)
    kept = clean.columns[start:-config.SHIFT_WIDTH]
    for col in kept:
        a = pd.to_numeric(clean[col], errors="coerce") if col not in config.SHIFT_TEXT_COLUMNS else clean[col]
        b = fixed[col]
        if col in config.SHIFT_TEXT_COLUMNS:
            assert (a.astype(str) == b.astype(str)).all(), col
        else:
            np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True,
                                       err_msg=col)


def test_truncated_tail_is_missing_and_clean_rows_untouched():
    broken = synthetic.generate(n_matches=3, seed=3)
    mask = repair.detect_shifted_rows(broken)
    fixed, _ = repair.repair_column_shift(broken)
    tail = fixed.columns[-config.SHIFT_WIDTH:]
    assert fixed.loc[mask.to_numpy(), tail].isna().all().all()
    # Columns before the shift point are never moved.
    start = list(broken.columns).index(config.SHIFT_START_COLUMN)
    pd.testing.assert_frame_equal(fixed.iloc[:, :start], broken.iloc[:, :start])


def test_tiny_hand_built_frame():
    cols = ["id", "PitchX", "B", "Ball Type", "C", "Pitchside Fielder 2 Y", "D"]
    clean_row = [1, 1.0, 2.0, "Kookaburra", 3.0, 9.0, 4.0]
    # Same delivery exported with 2 junk cells inserted at PitchX; the last 2 fall off.
    shifted_row = [2, -7.0, -7.0, 1.0, 2.0, "Kookaburra", 3.0]
    df = pd.DataFrame([clean_row, shifted_row], columns=cols, dtype=object)
    mask = repair.detect_shifted_rows(df)
    assert mask.tolist() == [False, True]
    fixed, rep = repair.repair_column_shift(df, mask=mask, width=2, text_columns=["Ball Type"])
    assert rep.rows_repaired == 1
    row = fixed.loc[1]
    assert (row["PitchX"], row["B"], row["Ball Type"], row["C"]) == (1.0, 2.0, "Kookaburra", 3.0)
    assert pd.isna(row["Pitchside Fielder 2 Y"]) and pd.isna(row["D"])
