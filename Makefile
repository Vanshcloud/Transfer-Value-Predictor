# Target names deliberately match the sibling predictive-maintenance project,
# so muscle memory carries between the two repositories.

.PHONY: help setup install install-dev hooks test test-cov lint format format-check typecheck quality clean

PYTHON := python3.13
VENV   := .venv
BIN    := $(VENV)/bin

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: ## Create the venv, install dev dependencies, install git hooks
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(MAKE) hooks
	@echo "Setup complete. On macOS, lightgbm and xgboost also need: brew install libomp"

hooks: ## Point git at the version-controlled hooks directory
	git config core.hooksPath scripts/hooks
	@echo "core.hooksPath = scripts/hooks"

install: ## Install runtime dependencies only
	$(BIN)/pip install -r requirements.txt

install-dev: ## Install runtime + dev dependencies
	$(BIN)/pip install -r requirements-dev.txt

test: ## Run unit tests (integration excluded)
	$(BIN)/pytest -m "not integration"

test-cov: ## Run tests with a coverage report
	$(BIN)/pytest -m "not integration" --cov=src --cov-report=term-missing

lint: ## Run ruff
	$(BIN)/ruff check src tests scripts

format: ## Format with black and apply ruff import order
	$(BIN)/black src tests scripts
	$(BIN)/ruff check --fix src tests scripts

format-check: ## Check formatting without writing
	$(BIN)/black --check src tests scripts

typecheck: ## Run mypy
	$(BIN)/mypy src

quality: lint format-check typecheck ## Run every quality gate

clean: ## Remove caches and build artefacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
