.PHONY: help install install-dev test lint format type-check clean

help:
	@echo "Agent Harness Development Commands"
	@echo ""
	@echo "  install       Install package"
	@echo "  install-dev   Install package with dev dependencies"
	@echo "  install-all   Install package with all providers"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage"
	@echo "  lint          Run linter (ruff)"
	@echo "  format        Format code with black"
	@echo "  format-check  Check code formatting"
	@echo "  type-check    Run type checker (mypy)"
	@echo "  clean         Remove build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-all:
	pip install -e ".[all,dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

lint:
	ruff check .

format:
	black .

format-check:
	black --check .

type-check:
	mypy agent_harness.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
