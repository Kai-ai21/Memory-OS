.PHONY: up down install test test-unit lint fmt typecheck check run

up:        ; docker compose up -d
down:      ; docker compose down
install:   ; pip install -e ".[dev]"
test:      ; pytest
test-unit: ; pytest -m "not integration"
lint:      ; ruff check .
fmt:       ; ruff format . && ruff check --fix .
typecheck: ; mypy
check: lint typecheck test
run:       ; uvicorn "memoryos.api.app:create_app" --factory --reload
