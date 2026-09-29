"""Detection and repair of the 2023/24 five-column export shift.

Background (dissertation Section 3.2.3): every row from the 2023 and 2024 seasons
was exported with five extra columns inserted part-way along the row. Every
value from that point on sits five columns to the right of its header, and the
final five values are truncated. The 2025 season was exported correctly, and
the two exports were concatenated under one header row.

Three independent fingerprints identify the affected rows:

* ``Ball Type`` text ("Kookaburra") appears in ``Pitchside Fielder 2 Y``.
* ``BounceSpeed`` / ``StumpSpeed`` / ``DropSpeed`` equal ``ReleaseX/Y/Z`` read
  five columns to the left.
* The boundary between affected and clean rows is the season boundary.

Because the displacement is uniform, the repair is exact: shift the tail of each
affected row five places left and set the last five cells to missing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class RepairReport:
    rows_repaired: int
    start_index: int
    width: int
    ball_type_restored: float
    sentinel_remaining: int
    median_check: pd.DataFrame


def detect_shifted_rows(
    df: pd.DataFrame,
    column: str = config.SHIFT_SENTINEL_COLUMN,
    value: str = config.SHIFT_SENTINEL_VALUE,
) -> pd.Series:
    """Boolean mask of rows carrying the sentinel text in a numeric column."""
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].astype(str).str.strip().eq(value)


def repair_column_shift(
    df: pd.DataFrame,
    mask: pd.Series | None = None,
    start_column: str = config.SHIFT_START_COLUMN,
    width: int = config.SHIFT_WIDTH,
    text_columns: list[str] = config.SHIFT_TEXT_COLUMNS,
) -> tuple[pd.DataFrame, RepairReport]:
    """Shift the tail of affected rows ``width`` places to the left.

    Parameters
    ----------
    df:
        Raw export, one row per delivery, header from the correctly exported season.
    mask:
        Rows to repair. Defaults to :func:`detect_shifted_rows`.
    start_column:
        Header of the first displaced column (position 105 in the real export).
    width:
        Number of inserted columns.

    Returns
    -------
    repaired frame and a :class:`RepairReport`.
    """
    out = df.copy()
    if mask is None:
        mask = detect_shifted_rows(out)
    mask = mask.to_numpy() if isinstance(mask, pd.Series) else np.asarray(mask)

    start = list(out.columns).index(start_column)
    if mask.any():
        values = out.to_numpy(dtype=object, copy=True)
        tail = values[mask, start + width:]
        values[mask, start:start + tail.shape[1]] = tail
        values[mask, start + tail.shape[1]:] = np.nan
        out = pd.DataFrame(values, columns=out.columns, index=out.index)

    for col in out.columns[start:]:
        if col not in text_columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Distribution check: repaired seasons should now match the clean season.
    check_cols = [c for c in ["SwingAngle", "DropAngle", "BounceAngle", "ImpactX"] if c in out]
    median_check = pd.DataFrame({
        "repaired_median": [out.loc[mask, c].median() for c in check_cols],
        "clean_median": [out.loc[~mask, c].median() for c in check_cols],
    }, index=check_cols)

    report = RepairReport(
        rows_repaired=int(mask.sum()),
        start_index=start,
        width=width,
        ball_type_restored=float(out.loc[mask, "Ball Type"].eq(config.SHIFT_SENTINEL_VALUE).mean())
        if mask.any() and "Ball Type" in out else 1.0,
        sentinel_remaining=int(detect_shifted_rows(out).sum()),
        median_check=median_check,
    )
    return out, report
