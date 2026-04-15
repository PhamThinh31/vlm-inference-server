.PHONY: dev test test-unit test-integration lint typecheck load-test bench fmt clean

VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

dev:
	python3.10 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements-dev.txt
	$(VENV)/bin/pre-commit install
	@echo "done. run: source $(VENV)/bin/activate"

test: test-unit test-integration

test-unit:
	$(VENV)/bin/pytest tests/unit -v

test-integration:
	$(VENV)/bin/pytest tests/integration -v

lint:
	$(VENV)/bin/ruff check app tests

fmt:
	$(VENV)/bin/ruff format app tests

typecheck:
	$(VENV)/bin/mypy app --ignore-missing-imports

# Sustained load against a running server. Assumes the API is up on
# :8000 with the test_images/ corpus mounted. Writes metrics JSON that
# can be diffed against stress_test_metrics.json for regressions.
load-test:
	$(VENV)/bin/python tests/stress_test_with_data.py \
		--concurrency 16 --duration 120 --out stress_test_metrics.json

# Single-shot latency snapshot. Fast — for quick before/after on a
# config change.
bench:
	$(VENV)/bin/python tests/quick_load_test.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache $(VENV)
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
