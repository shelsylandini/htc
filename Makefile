.PHONY: help install dev lint type test docker clean

help:
	@echo "install  install runtime dependencies"
	@echo "dev      install with dev extras and pre-commit"
	@echo "lint     ruff + black --check + isort --check"
	@echo "type     mypy --strict on htc_twin"
	@echo "test     pytest"
	@echo "docker   build the container image"
	@echo "clean    remove caches and build artefacts"

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	black --check .
	isort --check-only .

type:
	python -m mypy htc_twin

test:
	pytest

docker:
	docker build -t closedloop_oa:0.1.0 .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	find . -type d -name '.mypy_cache' -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	rm -rf runs
