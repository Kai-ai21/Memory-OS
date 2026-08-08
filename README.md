# Memory Intelligence OS

A long-term AI memory system. The goal is durable, queryable memory for AI agents — storing
what was learned, retrieving it by meaning rather than by keyword, and keeping it coherent as
it grows. Postgres 17 with `pgvector` is the storage substrate.

## Status

**Phase 1, Milestone 1.4 — normalization and chunking.**

Point it at a directory and it walks the tree, hashes every file, stores the bytes, records
artifacts and events, versions memories, parses each artifact into normalized text, and splits
that text into chunks sized for an embedding model.

It does not embed. Chunks land with `embedding IS NULL` and a `chunker_version` stamp; M1.5
fills the vectors.

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

## Normalization and chunking

Every artifact is parsed into one shape — plain text, a title, metadata, and the structural
markers the format already carried — so nothing downstream needs a branch per file type.
Markdown contributes headings, Python contributes top-level definitions via `ast`, PDFs
contribute page boundaries.

Text is then normalized: NFC, LF line endings, no trailing whitespace, runs of blank lines
collapsed, leading BOM removed. **`normalized_hash` is the second hash level and the point of
the whole step.** A file saved with CRLF endings is genuinely different bytes, so it is
genuinely a new artifact and a new memory version — but its normalized text is identical, so
its existing chunks simply move to the new version. No re-chunking, and in M1.5 no
re-embedding, because the vectors travel with the rows.

Chunking splits on structure first, because the author already said where the topic changes.
Oversized sections are filled sentence-aware; undersized ones merge with their neighbour; each
chunk carries an overlap prefix from the one before it, because boundaries are arbitrary and a
concept spanning one would otherwise appear in neither chunk in full. `char_start`/`char_end`
index exactly into the stored text, which is what will let a citation highlight the matched
span.

**Code is special-cased**: it splits only on definition boundaries and never mid-function, even
when that leaves a chunk over the ceiling. Half a function embeds as neither a function nor a
coherent statement; size variance is the lesser cost.

The chunker version encodes its parameters:

```
structural-v1:target=640:overlap=80:min=120:max=1024
```

which makes improving the chunker a query rather than a corpus rebuild:

```bash
memoryos rechunk --dry-run          # what is stale?
memoryos rechunk --source notes     # enqueue those, and only those
```

## Migrations

```bash
alembic upgrade head       # apply
alembic downgrade base     # unwind; implemented and tested
```

The URL comes from `Settings`, not from `alembic.ini`. Migrations are written by hand so that
constraint names and CHECK expressions are explicit; `alembic revision --autogenerate` is then
run as a check that the migration and the models agree, and must produce an empty revision.

## Ingestion

```bash
memoryos source add --kind filesystem --name notes --root ~/notes
memoryos source list
memoryos sync --source notes --full
```

A sync walks, hashes, stores, records, and enqueues. Nothing else — every source has a
different walking problem and an identical downstream pipeline, and keeping that line sharp is
what makes the second connector cheap.

**Re-running a sync is free.** If the artifact is already known and the current memory has the
same content hash, the item is skipped: no blob write, no artifact, no event, no version, no
job. That is what makes running it often reasonable.

**Change detection is two-tiered.** `(mtime, size)` decides *which* files are worth hashing;
the content hash decides whether anything actually changed. mtime alone is not trustworthy —
copying resets it, some editors preserve it, sync tools rewrite it — so it is a cheap filter
and never the authority.

**Deletion needs a full sweep.** An incremental sync cannot detect deletions: a deleted file
produces no observation at all, and absence is not an event. Only comparing the complete
observed set against the complete known set reveals it, which is why `--full` exists and why
`sources` carries `last_full_sync_at` separately from `last_sync_at`. A deletion records an
`ITEM_DELETED` event with `occurred_at_source = unknown`, because what we know is when we
noticed the absence, not when it happened.

Bytes live in a content-addressed blob store under `MEMOS_BLOB_ROOT` (default `./var/blobs`),
fanned out as `ab/cd/abcdef…`. Writes go to a temp file and are moved into place with
`os.replace`, so a crash can never leave a partial file at a path that claims to be a specific
hash.

## Endpoints

| Endpoint                   | Purpose                                                    |
| -------------------------- | ---------------------------------------------------------- |
| `/health/live`             | Liveness. Never touches external dependencies.              |
| `/health/ready`            | Readiness. Reports database connectivity and pgvector.      |
| `POST /sources`            | Register a source.                                          |
| `GET /sources`             | List registered sources.                                    |
| `POST /sources/{id}/sync`  | Enqueue a sync. Returns `202` with the job id.              |
| `GET /memories`            | Current memories, filterable by `source_id`, paginated.     |

`POST /sources/{id}/sync` enqueues and never runs the sync inline. A large directory takes
minutes to walk; doing it in the request would blow the HTTP timeout, and whatever it managed
would have no retry, no progress, and no way to resume.

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
