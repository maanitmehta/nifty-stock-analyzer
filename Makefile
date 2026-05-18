.DEFAULT_GOAL := help
VENV         := .venv
PYTHON       := $(VENV)/bin/python3
PIP          := $(VENV)/bin/pip
STREAMLIT    := $(VENV)/bin/streamlit
PYTEST       := $(VENV)/bin/pytest
RUFF         := $(VENV)/bin/ruff
BLACK        := $(VENV)/bin/black
MYPY         := $(VENV)/bin/mypy

.PHONY: help install install-dev test lint fmt typecheck run universe docker-build docker-run clean

help:
	@echo "NSE Stock Analyzer — available commands:"
	@echo ""
	@echo "  make install       Install runtime dependencies into .venv"
	@echo "  make install-dev   Install runtime + dev dependencies"
	@echo "  make run           Start the Streamlit dashboard"
	@echo "  make universe      Download full NSE equity list (~1800 stocks)"
	@echo "  make test          Run the test suite with coverage"
	@echo "  make lint          Run ruff linter"
	@echo "  make fmt           Auto-format with black"
	@echo "  make typecheck     Run mypy type checker"
	@echo "  make docker-build  Build the Docker image"
	@echo "  make docker-run    Run the Docker container on port 8501"
	@echo "  make clean         Remove .venv, caches, and build artefacts"

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install .

install-dev:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

run:
	$(STREAMLIT) run src/nifty_analyzer/ui/app.py

universe:
	$(PYTHON) scripts/download_universe.py

test:
	$(PYTEST) tests/ -v --cov=nifty_analyzer --cov-report=term-missing

lint:
	$(RUFF) check src/ tests/

fmt:
	$(BLACK) src/ tests/

typecheck:
	$(MYPY) src/nifty_analyzer

docker-build:
	docker build -t nifty-analyzer:latest .

docker-run:
	docker run --rm -p 8501:8501 \
		-v "$(PWD)/data/cache:/app/data/cache" \
		nifty-analyzer:latest

clean:
	rm -rf $(VENV) .mypy_cache .pytest_cache htmlcov coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
