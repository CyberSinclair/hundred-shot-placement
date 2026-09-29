"""Build notebooks/walkthrough.ipynb. Run, then execute with nbconvert."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# Walkthrough: shot-zone prediction on the synthetic sample

This notebook runs each stage of the pipeline step by step on the synthetic sample in `data/sample/`.

> **The data here is invented.** It has the same columns and the same data quality defects as the real Hampshire export, so the code has something realistic to work on, but the numbers below say nothing about The Hundred. The dissertation's real results are in the main README.

To keep it quick, the model uses 100 trees instead of 300."""),
code("""import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from shotzones import backtest, cleaning, config, model, ope, plots, repair, zones
from shotzones.features import check_cumulative_timing

pd.set_option("display.precision", 3)
raw = pd.read_csv("../data/sample/synthetic_raw.csv", low_memory=False)
raw.shape"""),
md("""## 1. Find and repair the column shift

On the 2023 and 2024 seasons, text that belongs in `Ball Type` turns up in a fielder coordinate column. That fingerprints rows where five extra columns were inserted during export."""),
code("""mask = repair.detect_shifted_rows(raw)
print(f"Shifted rows: {mask.sum():,} of {len(raw):,}")
raw.loc[mask, ["Date", "Ball Type", "Pitchside Fielder 2 Y", "Pitchside Fielder 3 X"]].head(3)"""),
code("""fixed, report = repair.repair_column_shift(raw)
print(f"Rows repaired: {report.rows_repaired:,}, sentinel values left: {report.sentinel_remaining}")
report.median_check  # repaired seasons should now look like the clean season"""),
md("""## 2. Check that running totals are recorded before the ball

If `Cumulative Batter Runs` already includes the current ball, it leaks the outcome. This check compares each change in the total with the runs off the current and previous ball."""),
code("""check_cumulative_timing(fixed, "Cumulative Batter Runs")"""),
md("## 3. Clean"),
code("""clean, log = cleaning.clean_deliveries(fixed)
log.to_frame()"""),
md("## 4. Learn shot zones from landing positions"),
code("""zm = zones.fit_zones(clean)
clean["shot_zone"] = zm.kmeans.labels_
zones.zone_profile(clean, centres=zm.centres()).round(2)"""),
code("""plots.zone_map(clean, zm.centres());"""),
md("## 5. Predict the zone, holding out one match at a time"),
code("""params = model.make_params(config.K_ZONES, n_estimators=100)
cv = model.lomo_cv(clean, params=params, verbose=False)
pd.Series(model.summarise(cv.y_true, cv.y_pred, cv.per_match)).round(3)"""),
code("""plots.confusion(cv.y_true, cv.y_pred, config.K_ZONES);"""),
md("""## 6. Backtest: is following the recommendation associated with more runs?

A delivery "followed" the recommendation when the batter hit into the predicted zone."""),
code("""d = cv.deliveries
backtest.followed_summary(d)"""),
code("""backtest.hypothesis_tests(d, n_perm=200)"""),
code("""backtest.breakdown(d, "phase", order=config.PHASE_ORDER)"""),
md("""## 7. Off-policy evaluation

The naive comparison mixes up decision quality with execution. The doubly-robust estimator values each recommended zone with a reward model fitted on other matches, then corrects it with propensity-weighted residuals."""),
code("""scored = ope.cross_fitted_rewards(ope.add_propensities(d, cv.proba))
tbl = ope.evaluate_policy(scored, by="phase", order=config.PHASE_ORDER, n_boot=1000)
tbl"""),
code("""plots.uplift_by_group(tbl, "Value of the most-likely-zone recommendation");"""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "notebooks/walkthrough.ipynb")
print("written")
