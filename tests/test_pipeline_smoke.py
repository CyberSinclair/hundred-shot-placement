import json

from shotzones import pipeline, synthetic


def test_pipeline_runs_end_to_end(tmp_path):
    data = tmp_path / "raw.csv"
    synthetic.generate(n_matches=4, seed=7).to_csv(data, index=False)
    summary = pipeline.run(data, tmp_path / "out", fast=True, skip_robustness=True)
    assert (tmp_path / "out" / "RESULTS.md").exists()
    saved = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert saved["repair"]["sentinel_remaining"] == 0
    assert 0 < summary["model"]["accuracy"] <= 1
    assert summary["model"]["n_matches"] == 4
