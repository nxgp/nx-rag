# Thin wrappers around the real tools. Targets are placeholders where the
# underlying scripts do not exist yet (they arrive with their milestones).
.DEFAULT_GOAL := help
.PHONY: help up down logs lint format typecheck test test-int eval security check

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	 awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Start infrastructure (Qdrant, MLflow)
	docker compose up -d --remove-orphans qdrant mlflow

down: ## Stop and remove infrastructure containers
	docker compose down --remove-orphans

logs: ## Tail infrastructure logs
	docker compose logs -f

lint: ## Lint
	ruff check src tests

format: ## Auto-format
	ruff format src tests

typecheck: ## Static type check
	mypy

test: ## Run unit tests
	pytest

check: ## Run lint, format, and unit tests in one command
	./scripts/check.sh

test-int: ## Run integration tests (needs `make up`)
	pytest -m integration

eval: ## Run the evaluation matrix (placeholder until Milestone 8)
	@echo "Not implemented yet — see docs/planning/roadmap.md (Milestone 8)."

security: ## Static security + dependency audit
	bandit -q -r src
	pip-audit
