.PHONY: up down install test test-unit test-slow lint fmt typecheck check run worker \
        phase1-check types web web-install test-web dev restore full-check

up:        ; docker compose up -d
down:      ; docker compose down
install:   ; uv sync --frozen --extra dev
test:      ; pytest
test-unit: ; pytest -m "not integration and not graph and not slow"
test-slow: ; pytest -m slow
lint:      ; ruff check .
fmt:       ; ruff format . && ruff check --fix .
typecheck: ; mypy
check: lint typecheck test
run:       ; uvicorn "memoryos.api.app:create_app" --factory --reload
worker:    ; memoryos worker

# --------------------------------------------------------------------------
# Web
# --------------------------------------------------------------------------

web-install: ; cd web && npm ci

# Regenerate the API types from the routes themselves. The schema is dumped from
# the app object rather than fetched from a URL, so this needs no server and no
# database — which is what lets CI run the same command and diff the result.
#
# Hand-written API types drift silently, and this project has twice paid for "two
# places that must agree with nothing checking".
types:
	uv run python scripts/dump_openapi.py > web/src/api/openapi.json
	cd web && npx openapi-typescript src/api/openapi.json -o src/api/schema.d.ts

web:      ; cd web && npm run dev
test-web: ; cd web && npm run test -- --run

# API and UI together, for actually using the thing. The UI talks to the API
# across an origin, so MEMOS_CORS_ORIGINS has to name it; that is the whole
# reason the setting exists.
#
# The trap kills the API when the foreground dev server exits, so Ctrl-C does not
# leave uvicorn holding port 8000.
dev:
	@MEMOS_CORS_ORIGINS='["http://localhost:5173"]' \
	  uv run uvicorn "memoryos.api.app:create_app" --factory --port 8000 & \
	  API=$$!; trap "kill $$API 2>/dev/null" EXIT INT TERM; \
	  cd web && npm run dev

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
# `--only` on the next line is what makes this target terminate.
#
# Embedding enqueues a Phase 3 entity extraction per memory. On a machine with an
# API key configured, an unrestricted drain here runs one live model call per
# file — hundreds of them, against a free tier that serves a few a minute — so a
# check about *ingestion* blocks for hours on work Phase 1 does not own. The
# excluded jobs stay pending, so a later unrestricted drain still runs them.
	uv run memoryos worker --drain --only normalize_memory,embed_memory
	@printf "\n=== stats: after ingestion ===\n"
	uv run memoryos stats
	@printf "\n=== doctor ===\n"
	-uv run memoryos doctor
	@printf "\n=== verify-replay: rebuild into a shadow schema and compare ===\n"
	uv run memoryos verify-replay
	@printf "\n=== replay: rebuild in place, recomputing every vector ===\n"
	uv run memoryos replay --from-beginning --clear-cache
	@printf "\n=== stats: after replay (must match the first) ===\n"
	uv run memoryos stats
	@printf "\n=== the four assessment queries ===\n"
	uv run memoryos search "how does the job queue claim work" -k 3
	uv run memoryos search "why do we store two timestamps" -k 3
	uv run memoryos search "content addressing and deduplication" -k 3
	uv run memoryos search "what happens when a file is deleted" -k 3
	@printf "\n=== doctor: after replay ===\n"
	-uv run memoryos doctor

# --------------------------------------------------------------------------
# Everything, from a destroyed volume (M8.2)
# --------------------------------------------------------------------------
#
# `report --full` is only worth showing somebody if it can be reproduced, and
# reproducing it means restoring the state that `docker compose down -v`
# destroys. Four tables cannot be rebuilt from the corpus, because nothing in
# the corpus contains them: the decisions somebody recorded, the outcomes
# somebody checked, the assumption verdicts somebody judged, and the relevance
# judgements that took an afternoon of clicking.
#
# All four have an export or a seed, and this target is the order they go back
# in. It is written down here rather than in a person's memory because the
# project's own verification recipe opens by destroying them, and a restore
# sequence nobody has run is a restore sequence that does not work.
#
# Entity extraction is last, needs an API key, and is bounded by `--limit`.
# Without a key the corpus is fully working for everything Phases 1, 2, 4 and 5
# do — `doctor` reports the absence as a note rather than a failure, which is the
# distinction it exists to draw. *With* one, extraction is a live model call per
# memory and the free tier this project uses serves a few a minute, so an
# unbounded run here is a target that does not finish in an afternoon. The limit
# makes the coverage number small and honest rather than absent; raise it, or run
# `memoryos extract-entities` on its own, when the quota is there.
restore: phase1-check
	@printf "\n=== the four tables no rebuild reproduces ===\n"
	uv run python scripts/restore_judgements.py
	uv run python scripts/seed_decisions.py
	uv run python scripts/seed_outcomes.py
	uv run python scripts/evaluate_assumptions.py
	@printf "\n=== extraction and the projection it feeds ===\n"
	-uv run memoryos extract-entities --limit 25
	-uv run memoryos resolve-entities
	-uv run memoryos graph rebuild
	@printf "\n=== the behavioural layers, over whatever that produced ===\n"
	-uv run memoryos patterns discover
	-uv run memoryos model derive

# The final proof: eight phases of accumulated machinery, from nothing.
full-check: restore
	uv run memoryos evaluate --k 10 --compare var/baseline.json
	uv run memoryos graph verify
	uv run memoryos verify-replay
	-uv run memoryos doctor
	uv run memoryos report --full
