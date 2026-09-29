"""XGBoost zone classifier and cross-validation (dissertation Sections 4.3 to 4.5).

The primary design is leave-one-match-out (LOMO): each match is held out in
turn, so the model is always scored on a match it has never seen, which is the
deployment condition. Grouped and random 5-fold are provided for the robustness
comparison in Section 5.4.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from . import config
from .features import FeatureBuilder, assert_no_leakage


def make_params(k: int, **overrides) -> dict:
    return {**config.XGB_PARAMS, "num_class": k, **overrides}


def class_weights(y: np.ndarray, scheme: str | None) -> np.ndarray | None:
    """None, 'sqrt' (inverse square-root frequency) or 'balanced' (Section 5.4.1)."""
    if scheme is None:
        return None
    if scheme == "balanced":
        return compute_sample_weight("balanced", y)
    if scheme == "sqrt":
        freq = pd.Series(y).value_counts()
        return pd.Series(y).map(np.sqrt(freq.max() / freq)).to_numpy()
    raise ValueError(f"Unknown weighting scheme: {scheme}")


@dataclass
class CVResult:
    """Out-of-fold predictions for every evaluated delivery."""

    deliveries: pd.DataFrame   # input rows plus predicted_zone and followed
    proba: np.ndarray          # (n_deliveries, k) out-of-fold probabilities
    per_match: pd.DataFrame    # accuracy and backtest summary per held-out match

    @property
    def y_true(self) -> np.ndarray:
        return self.deliveries["shot_zone"].to_numpy()

    @property
    def y_pred(self) -> np.ndarray:
        return self.deliveries["predicted_zone"].to_numpy()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, k: int, params: dict,
                 weighting: str | None, feature_kwargs: dict):
    fb = FeatureBuilder(**feature_kwargs).fit(train)
    X_tr, X_te = fb.transform(train), fb.transform(test)
    assert_no_leakage(X_tr.columns)
    y_tr = train["shot_zone"].to_numpy()
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr.to_numpy(), y_tr, sample_weight=class_weights(y_tr, weighting))
    proba = model.predict_proba(X_te.to_numpy())
    return proba, model, fb


def lomo_cv(
    df: pd.DataFrame,
    k: int = config.K_ZONES,
    params: dict | None = None,
    weighting: str | None = None,
    feature_kwargs: dict | None = None,
    min_match: int = config.MIN_MATCH_DELIVERIES,
    verbose: bool = True,
) -> CVResult:
    """Leave-one-match-out cross-validation.

    Matches with fewer than ``min_match`` cleaned deliveries are not evaluated
    (one real match met this criterion), but still contribute to training.
    """
    params = params or make_params(k)
    feature_kwargs = feature_kwargs or {}
    parts, probas, per_match = [], [], []
    matches = df["Match"].unique()
    for i, m in enumerate(matches, 1):
        te_mask = df["Match"].eq(m).to_numpy()
        if te_mask.sum() < min_match:
            continue
        train, test = df[~te_mask], df[te_mask].copy()
        proba, _, _ = _fit_predict(train, test, k, params, weighting, feature_kwargs)
        test["predicted_zone"] = proba.argmax(axis=1)
        test["followed"] = (test["shot_zone"] == test["predicted_zone"]).astype(int)
        parts.append(test)
        probas.append(proba)
        f = test["followed"].eq(1)
        per_match.append({
            "match": m,
            "n_test": int(te_mask.sum()),
            "accuracy": accuracy_score(test["shot_zone"], test["predicted_zone"]),
            "f1_weighted": f1_score(test["shot_zone"], test["predicted_zone"],
                                    average="weighted", zero_division=0),
            "followed_runs": test.loc[f, "Runs"].mean(),
            "not_followed_runs": test.loc[~f, "Runs"].mean(),
            "pct_followed": f.mean() * 100,
        })
        if verbose and (i % 10 == 0 or i == len(matches)):
            print(f"  LOMO fold {i}/{len(matches)}")
    deliveries = pd.concat(parts, ignore_index=True)
    pm = pd.DataFrame(per_match)
    pm["diff"] = pm["followed_runs"] - pm["not_followed_runs"]
    return CVResult(deliveries=deliveries, proba=np.vstack(probas), per_match=pm)


def kfold_cv(df: pd.DataFrame, k: int = config.K_ZONES, n_splits: int = 5,
             grouped: bool = True, params: dict | None = None,
             weighting: str | None = None) -> dict:
    """Grouped (by match) or random stratified k-fold, for the Section 5.4.2 check."""
    params = params or make_params(k)
    y = df["shot_zone"].to_numpy()
    if grouped:
        splits = GroupKFold(n_splits=n_splits).split(df, y, groups=df["Match"])
    else:
        splits = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=config.SEED).split(df, y)
    oof = np.full(len(df), -1)
    for tr, te in splits:
        proba, _, _ = _fit_predict(df.iloc[tr], df.iloc[te], k, params, weighting, {})
        oof[te] = proba.argmax(axis=1)
    return summarise(y, oof)


def summarise(y_true: Iterable, y_pred: Iterable, per_match: pd.DataFrame | None = None) -> dict:
    """Accuracy, F1 and the majority-class baseline (Table 5.4)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    baseline = pd.Series(y_true).value_counts(normalize=True).max()
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "majority_baseline": baseline,
        "random_baseline": 1 / len(np.unique(y_true)),
        "improvement_over_baseline": accuracy_score(y_true, y_pred) - baseline,
        "n_deliveries": int(len(y_true)),
    }
    if per_match is not None:
        out["mean_match_accuracy"] = per_match["accuracy"].mean()
        out["sd_match_accuracy"] = per_match["accuracy"].std()
        out["n_matches"] = int(len(per_match))
    return out


def per_class_report(y_true, y_pred) -> pd.DataFrame:
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    rows = {k: v for k, v in rep.items() if k.isdigit()}
    return pd.DataFrame(rows).T.rename_axis("zone")


def feature_importance(df: pd.DataFrame, k: int = config.K_ZONES,
                       params: dict | None = None) -> pd.DataFrame:
    """Gain importance from a model trained on all rows (Table 5.8)."""
    params = params or make_params(k)
    fb = FeatureBuilder().fit(df)
    X = fb.transform(df)
    model = xgb.XGBClassifier(**params).fit(X.to_numpy(), df["shot_zone"].to_numpy())
    imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = imp.index + 1
    imp["group"] = imp["feature"].map(_feature_group)
    return imp


def _feature_group(name: str) -> str:
    if name.startswith("Pitchside Fielder"):
        return "fielder"
    if any(name.startswith(p) for p in ("Bowler Type", "Line_", "Length_", "Delivery_")):
        return "delivery type"
    if name in config.TRACKING_FEATURES:
        return "ball tracking"
    return "game state"


def fielder_ablation(df: pd.DataFrame, season: int = 2025, k: int = config.K_ZONES,
                     params: dict | None = None, verbose: bool = False) -> pd.DataFrame:
    """LOMO within one season, with and without fielder coordinates.

    Restricting to 2025 removes the season signal that fielder presence would
    otherwise proxy (dissertation Section 6.3).
    """
    sub = df[df["season"] == season]
    rows = []
    for label, use in [("with fielders", True), ("without fielders", False)]:
        res = lomo_cv(sub, k=k, params=params, feature_kwargs={"use_fielders": use},
                      verbose=verbose)
        s = summarise(res.y_true, res.y_pred)
        rows.append({"model": label, "accuracy": s["accuracy"], "macro_f1": s["macro_f1"],
                     "n_deliveries": s["n_deliveries"]})
    return pd.DataFrame(rows)
