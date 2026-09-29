.PHONY: install data run run-fast test figures

install:
	pip install -e ".[dev]"

data:
	python -m shotzones.synthetic --matches 12 --out data/sample/synthetic_raw.csv

run:
	python -m shotzones.pipeline --out outputs

run-fast:
	python -m shotzones.pipeline --fast --out outputs

test:
	pytest

figures:
	python scripts/make_readme_figures.py
