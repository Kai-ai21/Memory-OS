# Memory Intelligence OS

A long-term AI memory system. The goal is durable, queryable memory for AI agents — storing
what was learned, retrieving it by meaning rather than by keyword, and keeping it coherent as
it grows. Postgres 17 with `pgvector` is the storage substrate.

## Status

**Phase 1, Milestone 1.2 — job queue and worker.**

There are still no product features. What exists is the application factory, the database
engine wiring, two health endpoints, the domain entities and value objects, the Postgres
schema with its migrations, the repository ports and their adapters, and a durable job queue
drained by a worker process. Nothing chunks, embeds, or ingests yet; the connector arrives in
M1.3 and will enqueue into this queue.

## Architecture

Four layers. Dependencies point **inward only**.

```
api/          → FastAPI routes and HTTP schemas. Thin, no business logic.
application/  → Use cases. Declares Protocol "ports" for what it needs.
domain/       → Entities, value objects, invariants. Pure Python, zero I/O.
adapters/     → Implements the ports (Postgres, pgvector, embedders, filesystem).
```

The rule: `adapters/` depends on `application/`, never the reverse. `domain/` imports nothing
from the other three layers.

Configuration is read once at the edge (`memoryos.config`). Nothing deeper in the stack reads
`os.environ`.

## Data model

Five tables. `ingestion_events` is append-only and is the source of truth; `memories` and
`memory_chunks` are projections that can be truncated and rebuilt from it.

| Table              | Holds                                                              |
| ------------------ | ------------------------------------------------------------------ |
| `sources`          | Where artifacts come from, plus opaque per-source sync cursors.     |
| `raw_artifacts`    | Content-addressed artifacts. The BLAKE2b-256 hash is the key.       |
| `ingestion_events` | Append-only log of everything observed. Replayed to rebuild state.  |
| `memories`         | One row per version of an item; one current version per item.       |
| `memory_chunks`    | Retrievable spans with offsets and a 384-dimension embedding slot.  |

Two design points carry the most weight:

**Every memory has two timestamps.** `occurred_at` is when the thing happened in the world;
`ingested_at` is when this system learned about it. A missing `occurred_at` is never filled in
with `ingested_at` — that would collapse every backfilled memory onto the day it was ingested.
Instead `occurred_at_source` records the provenance, and a CHECK constraint enforces that a
null `occurred_at` pairs with `unknown` and nothing else.

**Identity is content, not path.** `raw_artifacts` is keyed by hash, so re-reading an unchanged
file collides on the primary key and does nothing. Events reference hashes rather than paths,
because replay has to be deterministic and paths move.

Primary keys are UUIDv7 (`memoryos.domain.ids.new_id`). The embedded timestamp prefix keeps
inserts sequential in the B-tree instead of scattering them the way UUIDv4 does.

## Job queue

Work is queued in a `jobs` table and drained by a worker process.

```bash
make worker                      # run until SIGTERM
memoryos worker --drain          # exit once the queue is empty
```

A table rather than a broker, for two reasons.

**No dual-write problem.** Enqueueing a job and writing the data it refers to happen in one
transaction (`enqueue_in`). With a broker you can commit the row and fail to enqueue, or
enqueue and fail to commit; the standard fix for that is the transactional outbox pattern,
which is a jobs table in your database anyway.

**Observability is free.** `SELECT status, count(*) FROM jobs GROUP BY 1` is the whole
monitoring story, and every failure's error and traceback are queryable with SQL you already
know. The ceiling is low thousands of jobs per second — far above anything embedding will ask
for — and the `JobQueue` port can be swapped without touching a use case if that ever changes.

Three mechanics carry the design:

- **`FOR UPDATE SKIP LOCKED`** in the claim query. Without it every worker selects the same top
  row and all but one block on its lock, so the queue drains serially however many workers run.
- **`attempts` increments on claim, not on failure.** A worker that segfaults never reaches its
  failure handler, so a job that reliably kills the process would otherwise retry forever with
  `attempts` stuck at zero.
- **Fencing.** Every mutating call carries `AND locked_by = :worker_id` and returns a bool. A
  worker whose lease expired mid-handler cannot write over a job another worker now owns.

Retries back off exponentially with jitter. The jitter is not decoration: without it, a
thousand jobs that failed together retry together, and keep a recovering dependency down.

## Migrations

```bash
alembic upgrade head       # apply
alembic downgrade base     # unwind; implemented and tested
```

The URL comes from `Settings`, not from `alembic.ini`. Migrations are written by hand so that
constraint names and CHECK expressions are explicit; `alembic revision --autogenerate` is then
run as a check that the migration and the models agree, and must produce an empty revision.

## Endpoints

| Endpoint        | Purpose                                                          |
| --------------- | ---------------------------------------------------------------- |
| `/health/live`  | Liveness. Never touches external dependencies.                    |
| `/health/ready` | Readiness. Reports database connectivity and the pgvector version. |

Liveness stays dependency-free on purpose: a database blip should not cause an orchestrator to
kill otherwise-healthy containers. Readiness is the endpoint that checks dependencies, and
returns `503` when the database is unreachable or `pgvector` is not installed.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (dependency resolution and virtualenv management)
- Docker (for Postgres)

## Quickstart

```bash
make up        # start Postgres 17 + pgvector on host port 5433
make install   # uv sync --frozen --extra dev
make check     # ruff + mypy --strict + pytest
make run       # uvicorn on http://localhost:8000
```

Host port `5433` is deliberate, so the container does not collide with a local Postgres on
`5432`.

Copy `.env.example` to `.env` to override defaults. All settings use the `MEMOS_` prefix.

## Tests

```bash
make test-unit  # unit tests only, no database required
make test       # everything, including integration tests
```

Integration tests are marked `integration` and need Postgres running.

If `/health/ready` reports a null `pgvector_version`, the init script did not run because the
data volume already existed. Reset it:

```bash
docker compose down -v && docker compose up -d
```
