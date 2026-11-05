PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

.PHONY: report test lint check

report:
	$(PY) -m impact_lab.report --out reports

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .

# ruff + pytest (includes the report smoke test)
check: lint test
