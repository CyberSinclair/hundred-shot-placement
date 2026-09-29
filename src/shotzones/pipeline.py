"""End-to-end pipeline: raw export -> repaired -> cleaned -> zones -> model -> backtest -> OPE.

Usage::

    python -m shotzones.pipeline --data data/sample/synthetic_raw.csv --out outputs
    python -m shotzones.pipeline --data path/to/Mens_Hundred.xlsx --out outputs_real
    python -m shotzones.pipeline --fast   # fewer trees, permutations and bootstraps

Every stage writes its tables to ``--out`` and a short ``RESULTS.md`` summary is
written at the end.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # the pipeline only writes files

from . import backtest, cleaning, config, model, ope, plots, recommend, repair, zones  # noqa: E402
from .features import check_cumulative_timing  # noqa: E402


def load_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def _fmt(x, nd=3):
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) > 1):
        return f"{int(x):,}"
    return f"{x:.{nd}f}" if isinstance(x, float) and pd.notna(x) else str(x)


def run(data: Path, out: Path, fast: bool = False, k: int = config.K_ZONES,
        skip_robustness: bool = False) -> dict:
    t0 = time.time()
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    n_perm = 200 if fast else config.N_PERMUTATIONS
    n_boot = 1000 if fast else config.N_BOOTSTRAP
    params = model.make_params(k, **({"n_estimators": 100} if fast else {}))
    summary: dict = {"data": str(data), "fast": fast, "k": k}

    print("[1/7] Load and repair")
    raw = load_raw(data)
    repaired, rep = repair.repair_column_shift(raw)
    summary["repair"] = {"rows_repaired": rep.rows_repaired,
                         "ball_type_restored": rep.ball_type_restored,
                         "sentinel_remaining": rep.sentinel_remaining}
    rep.median_check.to_csv(out / "01_repair_median_check.csv")

    checks = []
    if {"Cumulative Batter Runs", "Runs"} <= set(repaired.columns):
        checks.append(check_cumulative_timing(repaired, "Cumulative Batter Runs"))
    if {"Team Runs", "Runs", "Extra Runs"} <= set(repaired.columns):
        ball_total = repaired.assign(
            _ball_total=pd.to_numeric(repaired["Runs"], errors="coerce").fillna(0)
            + pd.to_numeric(repaired["Extra Runs"], errors="coerce").fillna(0))
        checks.append(check_cumulative_timing(ball_total, "Team Runs", "_ball_total",
                                              group_cols=("Match", "Innings")))
    timing = pd.DataFrame(checks)
    timing.to_csv(out / "01_cumulative_timing_check.csv", index=False)

    print("[2/7] Clean")
    clean, log = cleaning.clean_deliveries(repaired)
    log.to_frame().to_csv(out / "02_cleaning_log.csv", index=False)
    summary["cleaning"] = {"final_rows": len(clean), "matches": int(clean["Match"].nunique())}

    print("[3/7] Shot zones")
    zm = zones.fit_zones(clean, k=k)
    clean["shot_zone"] = zm.kmeans.labels_
    kgrid = zones.k_search(clean)
    kgrid.to_csv(out / "03_k_search.csv", index=False)
    prof = zones.zone_profile(clean, centres=zm.centres())
    prof.to_csv(out / "03_zone_profile.csv")
    plots.zone_map(clean, zm.centres()).savefig(out / "figures" / "zone_map.png")

    print("[4/7] Leave-one-match-out cross-validation")
    cv = model.lomo_cv(clean, k=k, params=params)
    metrics = model.summarise(cv.y_true, cv.y_pred, cv.per_match)
    summary["model"] = metrics
    cv.per_match.to_csv(out / "04_per_match.csv", index=False)
    model.per_class_report(cv.y_true, cv.y_pred).to_csv(out / "04_per_class.csv")
    plots.confusion(cv.y_true, cv.y_pred, k).savefig(out / "figures" / "confusion.png")
    imp = model.feature_importance(clean, k=k, params=params)
    imp.to_csv(out / "04_feature_importance.csv", index=False)
    summary["fielder_gain_share"] = float(imp.loc[imp["group"] == "fielder", "importance"].sum())
    plots.importance(imp).savefig(out / "figures" / "feature_importance.png")

    print("[5/7] Backtesting")
    d = cv.deliveries
    backtest.followed_summary(d).to_csv(out / "05_followed_summary.csv", index=False)
    backtest.run_distribution(d).to_csv(out / "05_run_distribution.csv")
    tests = backtest.hypothesis_tests(d, n_perm=n_perm)
    tests.to_csv(out / "05_hypothesis_tests.csv", index=False)
    by_phase = backtest.breakdown(d, "phase", order=config.PHASE_ORDER)
    by_phase.to_csv(out / "05_by_phase.csv", index=False)
    d = d.assign(bowling=d["bowler_pace"].map({1: "Pace", 0: "Spin"}))
    backtest.breakdown(d, "bowling", order=["Pace", "Spin"]).to_csv(out / "05_by_bowling.csv", index=False)
    backtest.breakdown(d, "predicted_zone").to_csv(out / "05_by_zone.csv", index=False)
    fs = backtest.followed_summary(d)
    summary["backtest"] = {"mean_runs_followed": fs.loc[0, "mean_runs"],
                           "mean_runs_not_followed": fs.loc[1, "mean_runs"]}

    print("[6/7] Off-policy evaluation")
    scored = ope.add_propensities(cv.deliveries, cv.proba)
    scored = ope.cross_fitted_rewards(scored)
    ope_tbl = ope.evaluate_policy(scored, by="phase", order=config.PHASE_ORDER, n_boot=n_boot)
    ope_tbl.to_csv(out / "06_ope_by_phase.csv", index=False)
    summary["ope"] = ope_tbl.iloc[0][["behaviour", "DM", "IPS", "SNIPS", "DR", "DR_uplift",
                                      "uplift_ci_low", "uplift_ci_high"]].astype(float).to_dict()
    plots.uplift_by_group(ope_tbl, "Value of the most-likely-zone recommendation").savefig(
        out / "figures" / "ope_by_phase.png")

    ev = recommend.compare_policies(cv.deliveries, cv.proba, k=k, n_boot=n_boot,
                                    order=config.PHASE_ORDER)
    ev.to_csv(out / "06_policy_comparison.csv", index=False)

    if not skip_robustness:
        print("[7/7] Robustness checks")
        rob = pd.DataFrame([
            {"scheme": "Random 5-fold (delivery level)",
             **model.kfold_cv(clean, k=k, grouped=False, params=params)},
            {"scheme": "Grouped 5-fold (by match)",
             **model.kfold_cv(clean, k=k, grouped=True, params=params)},
            {"scheme": "Leave-one-match-out", **metrics},
        ])
        rob.to_csv(out / "07_cv_scheme_comparison.csv", index=False)
        if (clean["season"] == 2025).sum() > 200:
            model.fielder_ablation(clean, k=k, params=params).to_csv(
                out / "07_fielder_ablation_2025.csv", index=False)

    summary["runtime_seconds"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_results_md(out, summary, log, tests, by_phase, ope_tbl, ev, timing)
    print(f"Done in {summary['runtime_seconds']}s. Results in {out}/RESULTS.md")
    return summary


def _write_results_md(out, s, log, tests, by_phase, ope_tbl, ev, timing):
    m = s["model"]
    lines = [
        "# Pipeline results", "",
        f"Data: `{s['data']}`{'  (fast mode)' if s['fast'] else ''}", "",
        "> [!WARNING]",
        "> If this was run on the synthetic sample, these numbers describe invented data and",
        "> say nothing about The Hundred. The dissertation's results are in the main README.", "",
        "## Data repair", "",
        f"- Rows repaired: **{s['repair']['rows_repaired']:,}**",
        f"- Ball Type restored on repaired rows: {s['repair']['ball_type_restored']:.1%}",
        f"- Sentinel values remaining after repair: {s['repair']['sentinel_remaining']}", "",
        "### Running-total timing check", "",
        timing.to_markdown(index=False) if len(timing) else "_not run_", "",
        "## Cleaning", "", log.to_markdown(), "",
        "## Model (leave-one-match-out)", "",
        "| Metric | Value |", "|---|---:|",
        *[f"| {k.replace('_', ' ')} | {_fmt(v, 4)} |" for k, v in m.items()], "",
        "## Backtest hypothesis tests", "", tests.to_markdown(index=False, floatfmt=".4g"), "",
        "## Backtest by phase", "", by_phase.to_markdown(index=False, floatfmt=".3f"), "",
        "## Off-policy evaluation", "", ope_tbl.to_markdown(index=False, floatfmt=".3f"), "",
        "## Most-likely vs expected-value recommendation (sample-split)", "",
        ev.to_markdown(index=False, floatfmt=".3f"), "",
        "![Zones](figures/zone_map.png)", "![Confusion](figures/confusion.png)",
        "![Importance](figures/feature_importance.png)", "![OPE](figures/ope_by_phase.png)",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/sample/synthetic_raw.csv"))
    p.add_argument("--out", type=Path, default=Path("outputs"))
    p.add_argument("--k", type=int, default=config.K_ZONES)
    p.add_argument("--fast", action="store_true", help="Quick run for checking the code")
    p.add_argument("--skip-robustness", action="store_true")
    a = p.parse_args(argv)
    run(a.data, a.out, fast=a.fast, k=a.k, skip_robustness=a.skip_robustness)


if __name__ == "__main__":
    main()
