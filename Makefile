SHELL := /bin/bash
PYTHON ?= python

.PHONY: install install-all lint format test demo train-demo serve clean

install:
	$(PYTHON) -m pip install -e .

install-all:
	$(PYTHON) -m pip install -e ".[ml,api,dev]"

lint:
	ruff check .
	mypy src

format:
	ruff format .
	ruff check --fix .

test:
	pytest --cov=volley_forecast --cov-report=term-missing

demo:
	volley-forecast generate-demo-data --output data/sample/demo_athlete_matches.csv

train-demo:
	volley-forecast train --data data/sample/demo_athlete_matches.csv --model-dir artifacts/demo --config config/model.example.yaml

serve:
	volley-forecast serve --model-dir artifacts/demo --host 0.0.0.0 --port 8000

clean:
	rm -rf .cache artifacts .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
