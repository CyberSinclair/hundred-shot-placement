"""Sequential cleaning with an audit trail (dissertation Section 3.5, Table 3.1).

Each filter records how many rows it removed, so the cost of every decision is
visible. Rows are dropped in preference to imputation; the two exceptions
(Delivery and ImpactZ) are imputed later, inside each training fold, by
:mod:`shotzones.features`.
"""

from __future__ import annotations

import pandas as pd

from . import config


class CleaningLog:
    """Collects (step, rows remaining, rows removed) as filters are applied."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def record(self, step: str, before: int, after: int) -> None:
        self.rows.append({"step": step, "rows_remaining": after, "removed": before - after})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_markdown(self) -> str:
        lines = ["| Step | Rows remaining | Removed |", "|---|---:|---:|"]
        for r in self.rows:
            removed = "" if r["step"] == "Raw dataset" else f"{r['removed']:,}"
            lines.append(f"| {r['step']} | {r['rows_remaining']:,} | {removed} |")
        return "\n".join(lines)


def clean_deliveries(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningLog]:
    """Apply the cleaning pipeline and add the engineered columns."""
    log = CleaningLog()
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    for col in ["Innings", "Innings Ball", "Runs"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    log.record("Raw dataset", len(out), len(out))

    before = len(out)
    out = out[out["Legal Ball"].eq("Yes")]
    log.record("Legal deliveries only", before, len(out))

    for label, cols in config.DROPNA_STEPS:
        before = len(out)
        out = out.dropna(subset=[c for c in cols if c in out.columns])
        log.record(label, before, len(out))

    # Deliveries with no bat contact record where the ball came to rest behind
    # the batter (the keeper), not a shot placement (Section 3.2.2).
    before = len(out)
    out = out[~out["Connection"].isin(config.NOT_HIT_CONNECTIONS)]
    log.record("Remove not-hit deliveries", before, len(out))

    for label, cols in config.TRACKING_DROPNA_STEPS:
        before = len(out)
        out = out.dropna(subset=[c for c in cols if c in out.columns])
        log.record(label, before, len(out))

    out = add_engineered_columns(out)
    return out.reset_index(drop=True), log


def add_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Derived variables from Section 3.6 plus helpers used in evaluation."""
    out = df.copy()
    out["balls_remaining"] = 100 - out["Innings Ball"]
    out["is_second_innings"] = (out["Innings"] == 2).astype(int)
    out["is_powerplay"] = out["Power Play"].notna().astype(int)
    out["is_left_handed"] = out["Batting Hand"].eq("LHB").astype(int)
    out["bowler_pace"] = out["Bowler Type"].isin(config.PACE_TYPES).astype(int)
    out["season"] = out["Date"].dt.year
    out["phase"] = assign_phase(out["balls_remaining"])
    out["is_boundary"] = (out["Runs"] >= 4).astype(int)
    out["is_scoring"] = (out["Runs"] > 0).astype(int)
    if "Req Run Rate At Start" in out:
        out["Req Run Rate At Start"] = pd.to_numeric(out["Req Run Rate At Start"],
                                                     errors="coerce").fillna(0)
    return out


def assign_phase(balls_remaining: pd.Series) -> pd.Series:
    """Innings phase from balls remaining. Includes the 100th ball (see config)."""
    return pd.cut(balls_remaining, bins=config.PHASE_BINS, labels=config.PHASE_LABELS)
