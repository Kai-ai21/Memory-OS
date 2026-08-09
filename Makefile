.PHONY: up down install test test-unit test-slow lint fmt typecheck check run worker phase1-check

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

# Everything Phase 1 built, in one command, from an empty volume.
#
# Ordered so that each step can only pass if the previous one really worked: the
# corpus has to ingest before `doctor` can find nothing wrong with it, and it has
# to be intact before `verify-replay` can rebuild it and match. The two `stats`
# outputs bracket a full replay that recomputes every vector, and they have to
# agree — that pair is the milestone's headline result.
#
# `-` on doctor is deliberate: it exits non-zero on findings, and a finding here
# should be read rather than swallowed by a failed make target. verify-replay has
# no `-`, because a mismatch there is a real failure.
phase1-check:
	docker compose down -v
	docker compose up -d
	sleep 8
	uv run alembic upgrade head
	uv run alembic check
	uv run memoryos source add --kind filesystem --name self --root .
	uv run memoryos sync --source self --full
	uv run memoryos worker --drain
	@echo "\n=== stats: after ingestion ==="
	uv run memoryos stats
	@echo "\n=== doctor ==="
	-uv run memoryos doctor
	@echo "\n=== verify-replay: rebuild into a shadow schema and compare ==="
	uv run memoryos verify-replay
	@echo "\n=== replay: rebuild in place, recomputing every vector ==="
	uv run memoryos replay --from-beginning --clear-cache
	@echo "\n=== stats: after replay (must match the first) ==="
	uv run memoryos stats
	@echo "\n=== the four assessment queries ==="
	uv run memoryos search "how does the job queue claim work" -k 3
	uv run memoryos search "why do we store two timestamps" -k 3
	uv run memoryos search "content addressing and deduplication" -k 3
	uv run memoryos search "what happens when a file is deleted" -k 3
	@echo "\n=== doctor: after replay ==="
	-uv run memoryos doctor
