.PHONY: up down install test test-unit test-slow lint fmt typecheck check run worker

up:        ; docker compose up -d
down:      ; docker compose down
install:   ; uv sync --frozen --extra dev
test:      ; pytest
test-unit: ; pytest -m "not integration and not slow"
test-slow: ; pytest -m slow
lint:      ; ruff check .
fmt:       ; ruff format . && ruff check --fix .
typecheck: ; mypy
check: lint typecheck test
run:       ; uvicorn "memoryos.api.app:create_app" --factory --reload
worker:    ; memoryos worker
