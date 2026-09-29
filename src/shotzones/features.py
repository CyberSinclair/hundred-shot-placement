"""Feature construction with fold-aware imputation and an explicit allowlist.

Two changes from the original notebook, both aimed at leakage:

1. **Allowlist, not blocklist.** A column becomes a feature only if it is named in
   :mod:`shotzones.config`. The notebook kept every column not on an exclusion
   list, which let ``Extra Runs`` (an outcome of the ball) into the model.
2. **Imputation learned per training fold.** Medians and the modal Delivery type
   are computed on the training matches only and then applied to the held-out
   match.

It also provides :func:`check_cumulative_timing`, a quick test of whether a
running total (batter runs, team runs) is recorded before or after the ball.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config


def fielder_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c.startswith("Pitchside Fielder") and (c.endswith(" X") or c.endswith(" Y"))]


@dataclass
class FeatureBuilder:
    """Fit on training rows, transform any rows. Mirrors the sklearn API."""

    use_game_state: bool = True
    use_fielders: bool = True
    medians_: dict = field(default_factory=dict)
    delivery_mode_: str | None = None
    categories_: dict = field(default_factory=dict)
    columns_: list[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        self.columns_ = []
        self.medians_ = {c: float(pd.to_numeric(df[c], errors="coerce").median())
                         for c in config.MEDIAN_IMPUTE if c in df}
        if "Delivery" in df:
            self.delivery_mode_ = df["Delivery"].mode(dropna=True).iloc[0]
        self.categories_ = {c: sorted(df[c].dropna().astype(str).unique())
                            for c in config.CATEGORICAL_FEATURES if c in df}
        self.columns_ = list(self.transform(df.head(1)).columns)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        numeric = [c for c in config.TRACKING_FEATURES if c in df]
        if self.use_game_state:
            numeric += [c for c in config.GAME_STATE_FEATURES if c in df]
        numeric += [c for c in config.ENGINEERED_FEATURES if c in df]
        num = df[numeric].apply(pd.to_numeric, errors="coerce")
        for c, med in self.medians_.items():
            if c in num:
                num[c] = num[c].fillna(med)
        parts.append(num)

        yn = pd.DataFrame(index=df.index)
        for c in config.YES_NO_FEATURES:
            if c in df:
                yn[c] = df[c].map({"Yes": 1, "No": 0}).astype(float)
        parts.append(yn)

        for c, cats in self.categories_.items():
            col = df[c]
            if c == "Delivery" and self.delivery_mode_ is not None:
                col = col.fillna(self.delivery_mode_)
            col = pd.Categorical(col.astype(str), categories=cats)
            parts.append(pd.get_dummies(col, prefix=c, dtype=float).set_index(df.index))

        if self.use_fielders:
            fcols = fielder_columns(df)
            parts.append(df[fcols].apply(pd.to_numeric, errors="coerce")
                         .fillna(config.FIELDER_SENTINEL))

        X = pd.concat(parts, axis=1)
        if self.columns_:
            X = X.reindex(columns=self.columns_, fill_value=0.0)
        return X.fillna(config.FIELDER_SENTINEL).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def assert_no_leakage(columns) -> None:
    """Raise if any outcome column, or a dummy derived from one, is a feature."""
    bad = [c for c in columns
           if c in config.LEAKAGE_COLUMNS
           or any(c.startswith(f"{leak}_") for leak in config.LEAKAGE_COLUMNS)]
    if bad:
        raise ValueError(f"Outcome columns found in feature matrix: {bad}")


def check_cumulative_timing(
    df: pd.DataFrame,
    cumulative_col: str = "Cumulative Batter Runs",
    increment_col: str = "Runs",
    group_cols: tuple[str, ...] = ("Match", "Innings", "Batter ID"),
    order_col: str = "Innings Ball",
) -> pd.Series:
    """Is a running total recorded before or after the ball?

    For consecutive deliveries faced by the same batter, compute the change in
    the running total. If it equals the runs off the *current* ball the total is
    post-ball and leaks the outcome. If it equals the runs off the *previous*
    ball the total is pre-ball and is a legitimate feature.

    Run this on the raw export (legal and illegal balls) before any filtering,
    so consecutive rows really are consecutive.

    Returns a Series with the share of rows consistent with each reading and a
    verdict.
    """
    d = df.sort_values(list(group_cols[:-1]) + [order_col])
    d = d[[*group_cols, order_col, cumulative_col, increment_col]].copy()
    d[cumulative_col] = pd.to_numeric(d[cumulative_col], errors="coerce")
    d[increment_col] = pd.to_numeric(d[increment_col], errors="coerce")
    g = d.groupby(list(group_cols), sort=False)
    delta = g[cumulative_col].diff()
    prev_inc = g[increment_col].shift(1)
    valid = delta.notna()
    post = float((delta[valid] == d.loc[valid, increment_col]).mean())
    pre = float((delta[valid] == prev_inc[valid]).mean())
    verdict = ("post-ball (leaks the outcome)" if post > pre + 0.05
               else "pre-ball (safe)" if pre > post + 0.05 else "inconclusive")
    return pd.Series({"column": cumulative_col, "pre_ball_share": pre,
                      "post_ball_share": post, "rows_checked": int(valid.sum()),
                      "verdict": verdict})


def to_pre_delivery(
    df: pd.DataFrame,
    cumulative_col: str,
    group_cols: tuple[str, ...] = ("Match", "Innings", "Batter ID"),
    order_col: str = "Innings Ball",
) -> pd.Series:
    """Convert a post-ball running total to its pre-ball value (lag by one)."""
    d = df.sort_values(list(group_cols[:-1]) + [order_col])
    lagged = d.groupby(list(group_cols), sort=False)[cumulative_col].shift(1).fillna(0)
    return lagged.reindex(df.index)
