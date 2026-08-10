# Memory Intelligence OS

A long-term AI memory system. The goal is durable, queryable memory for AI agents — storing
what was learned, retrieving it by meaning rather than by keyword, and keeping it coherent as
it grows. Postgres 17 with `pgvector` is the storage substrate.

## Status

**Phase 1 complete**, plus M2.0a (the search interface) and M2.0 (the evaluation harness).

Point it at a directory and it walks the tree, hashes every file, stores the bytes, records
artifacts and events, versions memories, parses each artifact into normalized text, splits that
text into chunks sized for the embedding model, embeds them, and answers questions about them by
meaning. Then it can throw all of that away and rebuild it from the log, and prove the result is
identical.

Semantic search only — BM25, fusion, reranking, and synthesis are Phase 2. What it retrieves
is now measured rather than assumed: see [Evaluation](#evaluation).

### What Phase 1 built

| Milestone | Delivered                                                                     |
| --------- | ----------------------------------------------------------------------------- |
| **M1.0**  | Repository skeleton, four-layer architecture, health endpoints, CI.            |
| **M1.1**  | Data model and migrations. Content addressing, two timestamps, UUIDv7 keys.    |
| **M1.2**  | Durable job queue in Postgres, and a worker with leases, fencing and backoff.  |
| **M1.3**  | Filesystem connector, content-addressed blob store, two-tier change detection. |
| **M1.4**  | Parsers, text normalization, structural chunking, `normalized_hash`.           |
| **M1.5**  | Embedding pipeline with a content-addressed vector cache.                      |
| **M1.6**  | HNSW index, `VectorStore` port, `/search`, and recall measurement.             |
| **M1.6.1**| Hotfix: chunk sizes derived from the model's real window, counted properly.    |
| **M1.7**  | Replay. Rebuild every derived table from the log, and verify it byte for byte. |
| **M2.0a** | Search UI, judgement capture, and the golden set M2.0 is measured against.     |

The single most valuable thing Phase 1 produced is not a feature. It is that M1.6.1 exists: a
silent 89% truncation rate that every test passed, found only because somebody measured. Most of
the machinery below — the startup assertion, `doctor`, `eval-recall`, `verify-replay` — is there
to make that class of defect loud next time.

### End to end from a clean checkout

```bash
docker compose up -d && sleep 8      # Postgres 17 + pgvector on host port 5433
uv sync --frozen --extra dev
uv run alembic upgrade head

uv run memoryos source add --kind filesystem --name self --root .
uv run memoryos sync --source self --full
uv run memoryos worker --drain

uv run memoryos stats
uv run memoryos doctor
uv run memoryos search "how does the job queue claim work" -k 5
uv run memoryos verify-replay
```

Or in one command, which is the same sequence plus a fresh volume and both replay modes:

```bash
make phase1-check
```

First run downloads ~90MB of model weights into `MEMOS_HF_HOME` (default `./var/hf`), and the
virtualenv is roughly 900MB because of `torch`.

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

Seven tables, split into two groups that matter more than the count. `ingestion_events`,
`raw_artifacts` and `sources` are the source of truth and are never truncated. `memories`,
`memory_chunks`, `jobs` and `embedding_cache` are derived, and M1.7 rebuilds them from the first
group plus the blob store. The split is declared as data in
[`application/replay.py`](src/memoryos/application/replay.py), and a test fails if a new table is
not classified.

| Table              | Holds                                                              |
| ------------------ | ------------------------------------------------------------------ |
| `sources`          | Where artifacts come from, plus opaque per-source sync cursors.     |
| `raw_artifacts`    | Content-addressed artifacts. The BLAKE2b-256 hash is the key.       |
| `ingestion_events` | Append-only log of everything observed. Replayed to rebuild state.  |
| `memories`         | One row per version of an item; one current version per item.       |
| `memory_chunks`    | Retrievable spans with offsets and a 384-dimension embedding slot.  |
| `jobs`             | Durable work queue. Derived: a rebuild empties it.                  |
| `embedding_cache`  | Vectors keyed by (model, role, text). Content-addressed memoisation. |

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
concept spanning one would otherwise appear in neither chunk in full.

**`char_start`/`char_end` bound the chunk's own span, not the text stored for it.** M1.4
documented them as indexing exactly into the stored text, which holds only at ordinal 0: the
spans tile the document contiguously, while `content` additionally carries the borrowed overlap
head. `prefix_chars` records how long that head is, so the relationship is exact and stated
rather than rediscovered:

```
content[prefix_chars:] == memory.content[char_start:char_end]
```

The UI had to measure the corpus to work out what the offsets meant, which is what a derived
value nobody records costs. 28% of stored chunk text is borrowed lead-in, so reading the offsets
as bounds on `content` mis-highlights most chunks — plausibly, since the text it points at is
real text from the same document.

**Code is special-cased**: it splits on definition boundaries, and an oversized definition is
broken at the outermost boundary inside it — blank lines first, then statement starts at the
shallowest indentation the body uses, and never inside an unbalanced bracket, because a break
inside an open call splits one statement into two fragments. The ceiling still wins over all of
it: a single statement longer than the window is split at a line start inside the call rather
than left for the model to truncate.

The chunker version encodes its parameters:

```
structural-v3:model_window=512:target=396:overlap=47:min=79:max=496
```

which makes improving the chunker a query rather than a corpus rebuild:

```bash
memoryos rechunk --dry-run          # what is stale?
memoryos rechunk --source notes     # enqueue those, and only those
```

## Embeddings

```bash
memoryos embed --dry-run              # what is unembedded?
memoryos reembed --model NEW_ID       # after a model change
memoryos stats                        # coverage and cache size
```

`BAAI/bge-small-en-v1.5` runs locally: 384 dimensions to match the column fixed in M1.1, and a
**512-token window**. Vectors are unit length, which makes cosine similarity and inner product
the same number and lets M1.6 use the cheaper one.

**Chunk sizes are derived from that window, not chosen**, and counted with the model's own
tokenizer. M1.4 targeted 640 heuristic words against a model that read 256 WordPieces, so 89%
of chunks were silently truncated before embedding — distinct chunks sharing a prefix embedded
identically and search returned confident nonsense. The composition root now refuses to start
if the chunker can emit more tokens than the model reads, and `memoryos doctor` reports the
same condition for data already written. Embedding runs on a thread — it is CPU-bound matrix multiplication, and on the
event loop it would stall every other coroutine including health checks.

**Queries and passages are embedded through separate calls**, `embed_query` and `embed_passage`.
There is deliberately no role-free `embed` left: several retrieval models are asymmetric, and one
used symmetrically does not fail — it just retrieves worse than it should, which is the same shape
of silent defect as M1.6.1.

Whether bge's documented query instruction is actually *applied* is a measured decision, not an
assumed one, because the model card for v1.5 says to make it one: "we improve its retrieval ability
when not using instruction … the best method to decide whether to add instructions for queries is
choosing the setting that achieves better performance on your task." Measured on this corpus, it
loses. On the six-document fixture the prefix takes the queue question from `+0.0060` to `-0.0007`
— an *inverted* ranking, not a narrower margin — while improving the baking question from `+0.1344`
to `+0.1556`. On this repository's 719 chunks it changes the top result for two of the four
assessment queries, and costs "why do we store two timestamps" the file where the answer is
actually written. Mean margin prefers the prefix by 0.007; rankings reject it 1–2, and a ranking is
what a user sees. So the string is recorded in `DOCUMENTED_QUERY_PREFIXES` and `APPLY_QUERY_PREFIX`
is empty. `tests/slow/test_query_prefix.py` re-runs that A/B against whichever setting is
configured and fails if it stops being the better one.

**The model id and the role are both part of the cache key, and that is a correctness requirement
rather than an optimisation.** Keying on text alone would let a model upgrade silently reuse the
old model's vectors. Nothing would error; the index would simply hold two incompatible coordinate
systems, and similarity between them is arithmetically valid and semantically meaningless. The role
is there for the same reason one level down: without it, the same sentence embedded as a query and
as a passage collide on one entry and whichever ran second receives the other's vector.

The cache is its own table so that identical text in different memories is embedded once, and
so that a crash between embedding and the chunk update costs nothing — the retried job finds
every vector in cache and never touches the model.

Combined with M1.4's chunk adoption, a cosmetic edit is free end to end: a line-ending change
produces a new artifact and a new memory version, but the chunk rows move to it carrying their
vectors, so nothing is re-chunked and nothing is re-embedded.

The first install is large — `torch` and `sentence-transformers` bring the virtualenv to
roughly 900MB, and the model weights are a further ~90MB downloaded on first use into
`MEMOS_HF_HOME` (default `./var/hf`).

```bash
memoryos doctor   # oversized chunks, stale models, unembedded or empty memories
make test-slow    # the tests that load the real model
```

That test is the only thing standing between this pipeline and one that fills the column with
plausible-looking garbage: every other test runs against a deterministic fake, which cannot
tell you whether the vectors mean anything.

## Search

```bash
memoryos search "how does the job queue claim work" -k 5
memoryos search "..." --exact          # sequential scan, to see what the index missed
memoryos eval-recall --queries 50 --ef-search 40,100,200,400
```

```
GET  /search?q=...&k=10&source=NAME&kind=note&after=...&before=...
POST /search                            # same, for long queries
```

Chunks are what match; memories are what come back, each carrying the chunks that matched it —
with their char offsets and the metadata the chunker recorded, so a result can say *which
function* a span came from rather than only which file. That metadata is computed during
normalization and persisted, because the moment it is needed is query time, in a different process.
A memory scores as its **best** chunk, not its mean — a long document with one perfectly
relevant paragraph should outrank a short one that is vaguely on-topic throughout. Ties break
on the mean.

The index uses `vector_ip_ops` because the embedder normalizes: for unit vectors, inner
product and cosine similarity are the same number and inner product skips a division. The
adapter refuses to construct against a non-normalizing embedder, because that combination does
not error — it silently ranks wrongly.

`ef_search` is set with `SET LOCAL` per query. A session-level `SET` would ride the pooled
connection into every later query on it.

**Measure before tuning.** `eval-recall` samples chunks, uses each as its own query, and
compares the index against an exhaustive scan across several `ef_search` values. Measured on a
20,000-chunk corpus: recall@10 rises 0.94 → 1.00 as `ef_search` goes 40 → 400, and p50 latency
rises 2.2ms → 6.3ms. Below roughly 2,000 chunks Postgres correctly ignores the index
altogether and scans, so the numbers there say nothing about HNSW.

## Search interface

```bash
make dev     # API on :8000 with CORS for the UI, and Vite on :5173
make web     # just the UI
make types   # regenerate web/src/api/schema.d.ts from the routes
make test-web
```

A local web UI for searching the corpus, reading results with the matched span
highlighted, and capturing the judgements that seed M2.0's golden query set. Vite +
React + TypeScript + Tailwind, TanStack Query for server state, no component library.
Every piece of state in it is server state; the client state is a search box and some
filters, and those live in the URL so a search is linkable and the back button works.

**The API types are generated, not written.** `make types` dumps the OpenAPI schema
from the app object — no server, no database — and regenerates
`web/src/api/schema.d.ts`. CI fails if the committed types differ. This project has
twice paid for two places that must agree with nothing checking.

**CORS is off unless you name an origin.** `MEMOS_CORS_ORIGINS` defaults to empty, and
a wildcard raises at startup: this API answers questions about a private corpus, so
`*` means any page you visit can read it.

### The highlight

The most important element in the interface, and the one that required working out
what the stored offsets actually mean. `char_start`/`char_end` tile the document
contiguously — chunk N ends exactly where N+1 begins — but `chunk.content` is *longer*
than that span, because the chunker prepends an overlap prefix borrowed from the
previous chunk. Nothing records how long the prefix is. It is recoverable exactly:

```
prefix = len(content) - (char_end - char_start)
content == document[char_start - prefix : char_end]
```

verified for 934 of 934 chunks. So the chunk's own text is the *tail* of what is
stored, the borrowed context is the head, and **28.1% of all stored chunk text in this
corpus is duplicated from a neighbour**. The UI marks the tail and mutes the head, and
needs no extra request to do it.

### Judgements

Three verdicts per result — relevant, not relevant, missing — one click each. `missing`
is separate because by definition its subject is not on screen, and it carries no rank.
The same three buttons appear on each matched chunk, which is the only way to record
"right file, wrong chunk": M2.0 found `why do we store two timestamps` returning both
`README.md` and `models.py` inside the top six on paragraphs that do not distinguish
`occurred_at` from `ingested_at`, a failure a memory-level verdict scores as a success.

They land in `query_judgements`, the first table here that no machine can regenerate,
classified `USER_AUTHORED`: never truncated, never written by a replay. Identity is
`(source_name, external_key, chunk_ordinal)` rather than a memory id — with a null
ordinal meaning "this memory, whichever chunk matched" — and that is forced rather than
stylistic — a replay recreates every memory with a new UUID, and `TRUNCATE ... CASCADE`
empties any table referencing `memories`, so a foreign key there would mean every
rebuild destroyed the golden set. `memory_id`, `rank_at_judgement` and
`score_at_judgement` are snapshots of what the system said when a human disagreed with
it; the export re-resolves the natural key against the current corpus. The ordinal is
identity rather than a snapshot because, unlike `chunk_id`, it survives a rebuild —
chunking is deterministic, so chunk 4 of a file is chunk 4 again after a replay.

`NULLS NOT DISTINCT` on the unique key is load-bearing. Postgres treats nulls as
distinct by default, so once the ordinal joined the key, every memory-level row would
have stopped colliding with itself and the upsert that makes re-judging *replace* would
have quietly started appending instead.

```bash
memoryos export-golden-set --output var/golden-set.json
# and back again, because `docker compose down -v` destroys the only copy
uv run python scripts/restore_judgements.py var/golden-set.json
```

Demonstrated end to end: 56 judgements survived the corpus being destroyed and rebuilt
from the log with entirely new ids, and re-resolved with zero unresolved.

## Evaluation

Retrieval quality as a number, so later milestones can be argued about with evidence
rather than impressions.

```bash
memoryos evaluate --k 10 --json var/baseline.json
memoryos evaluate --k 10 --compare var/baseline.json   # what a change did
memoryos evaluate --query "SKIP LOCKED" --verbose      # why one query is bad
```

Four metrics, because each is blind to something the next one sees. **recall@k** —
of what should have been found, how much was; says nothing about order. **precision@k**
— of what came back, how much was relevant; penalises the noise recall rewards.
**MRR** — how high the *first* right answer ranked, which is the number that matters
when somebody reads one result and stops. **nDCG@k** — the whole ordering, discounted
by position. `application/metrics.py` is pure functions with no I/O and no knowledge of
this corpus; `application/golden.py` turns an export into scoreable queries.

`missing` counts as relevant — it is the only verdict naming something the ranking
failed to return, and without it recall would mean "of the things we found, how many
did we find". A query with no relevant judgements is **excluded with a warning** rather
than scored zero, and every triple is resolved against the current corpus at load time
with the failures *named*: a golden set that quietly shrinks as files move reports a
rising score for a corpus that is losing its answer key.

The baseline is 21 queries over 235 judgements at k=10, and it lives in
`var/baseline.json` — the four means are deliberately **not** copied into this file.

That is not laziness, it is the corpus. This repository is what gets indexed, so a
paragraph quoting the score is part of what produces the score, and keeping the two in
agreement is a fixpoint problem rather than an edit. It is not a rounding-scale effect
either: one draft of the section below happened to put a rare token next to the file
that contains it, inside one chunk, which put this README into that query's top ten and
moved the mean of every metric by about 0.015 — while the retrieval defect it was
describing was completely unchanged. Numbers that move when you write about them do not
belong in prose. `evaluate --compare` against the committed JSON is the interface, and
it is only meaningful over a corpus that did not change underneath the run.

**The worst-queries section is the useful output.** The mean says whether something
improved; the worst list says what to fix. The bottom of it is stable across runs:
`SKIP LOCKED`, `what stops the same document being stored twice`, and `how do I run the
worker`. The first two are one defect approached from opposite sides — a query whose
answer is a rare literal token the model reads as ordinary English, and a query whose
answer is phrased entirely in words the code never uses. Between them they are the case
for M2.1's lexical half.

The second failure the chunk ordinal exposes: `why do we store two timestamps` takes a
perfect MRR and loses two fifths of its recall, because `README.md` and `models.py` are
both inside the top six on paragraphs that never mention `ingested_at`. Judged per
memory that query looks solved.

`var/baseline.json` is committed. It is the record of where Phase 2 started, and every
later milestone in this phase reports its `--compare` diff against it.

## Replay

```bash
memoryos verify-replay                  # rebuild into a shadow schema and compare; exits non-zero on drift
memoryos replay --from-beginning        # rebuild the derived tables in place
memoryos replay --from-beginning --clear-cache   # and recompute every vector
memoryos replay --stage embed           # keep the chunks, redo only the vectors
memoryos replay --source notes          # one connector's corpus
memoryos replay --since 40000           # only events after a log position
memoryos replay --into-shadow           # build alongside the live tables, then swap
```

Every milestone before this one asserted in a docstring that the event log was the source of
truth. This is where that is either true or it is not.

**`--stage embed` is the one you actually reach for.** Swap the model, keep the chunks, recompute
the vectors: minutes rather than a full pipeline run. It works because the chunker version and the
model id are recorded per chunk, and because a chunk's identity owes nothing to its vector — the
rows and their ids survive, only the embeddings change.

**The cache is kept by default.** Truncating it makes a replay honest, because every vector is
recomputed; keeping it makes the replay fast but only proves the pipeline downstream of embedding.
Keeping it is correct rather than merely convenient — an entry is a pure function of
`(model, role, text)` — so `--clear-cache` is the stronger periodic check rather than the everyday
one.

**Nothing derived reads the clock.** `ingested_at` and `deleted_at` come from the causing event's
`recorded_at`, versions come from the order of the log, and the memory itself comes from
`projection.memory_from_event` — the same function `sync` calls, so "exactly as sync would have"
is structural rather than a promise. `deleted_at` used to come from `now()`; it was within
milliseconds of the event on the original write, which is precisely why it looked fine and would
have made every rebuild differ.

The one exception is `memory_chunks.embedded_at`, which records when a vector was computed rather
than anything about the data. It legitimately differs after a recomputation and is not compared.

**The pipeline runs interleaved, per event, in log order** — apply an event, normalize and embed
what it produced, then move on. An earlier version applied all the events and then normalized the
survivors, which is cheaper and wrong: a superseded version had been normalized before it was
superseded, and a deleted one had been normalized *and embedded* before it was tombstoned. That
version left `normalized_hash` null on every historical row and dropped the tombstoned version's
chunk. Neither is cosmetic — `normalize` finds a previous version *by its normalized hash* in
order to move chunks onto a cosmetically-changed file — and neither changes a single row count.
The integration test for versions and tombstones is what caught it.

The cost of that is worth stating plainly: an item with fifty revisions is parsed, chunked and
embedded fifty times, because that is what the log says happened. A replay is as expensive as the
history it replays.

### Verification

`verify-replay` snapshots the live derived state, rebuilds into a shadow schema, compares, and
drops the shadow. The live tables are never touched, so it is safe to run against a corpus
somebody is using.

Two rules decide how the comparison is written:

- **Content, not counts.** Counts match while `is_current` sits on the wrong version, versions are
  numbered in the wrong order, or vectors came from a different model. The M1.6.1 defect passed
  every count-based check ever written.
- **Natural keys, not primary keys.** Ids are UUIDv7, minted at write time, and a rebuild
  legitimately produces different ones. A memory is `(source, external_key, version)`; a chunk is
  that plus `ordinal`. Embeddings are compared as a digest of their packed bytes rather than
  formatted text, because a text rendering rounds and would let a changed model pass.

A verification that cannot fail is not a verification, so one test corrupts a chunk's hash — a
change invisible to every count — and requires the command to notice.

### The shadow swap, and what did not work

The design this was specified as does not work, and it is worth writing down because it looks
like it does:

```sql
BEGIN;
ALTER SCHEMA public RENAME TO memoryos_old;
ALTER SCHEMA memoryos_shadow RENAME TO public;
COMMIT;
```

Postgres stores every object reference by OID, not by name, so renaming a schema moves no
references. Measured on this database: a shadow table whose `source_id` referenced
`public.sources` referenced `memoryos_old.sources` after the swap, and the new `public` contained
no `sources` table at all — because the source of truth is deliberately never copied into the
workspace. The result is a live schema missing half its tables, with foreign keys reaching into
the schema you were about to drop. Dropping it then fails, or with `CASCADE` takes the constraints
and the referenced data with it. Every count-based check would pass.

What is implemented instead keeps the separate schema and moves the tables individually:

```sql
BEGIN;
DROP TABLE public.memory_chunks;                        -- child first
DROP TABLE public.memories;
ALTER TABLE memoryos_shadow.memories SET SCHEMA public; -- parent first
ALTER TABLE memoryos_shadow.memory_chunks SET SCHEMA public;
COMMIT;
```

`SET SCHEMA` moves the table itself, so its foreign keys keep pointing at `public.sources`.
Constraint and index names come across unchanged, and they can be the *canonical* names because
names are unique per schema rather than per database — the shadow `memories` carries `pk_memories`
while the live one still exists. That is what leaves the schema identical to the models, so
`alembic check` stays clean and the next replay still works. Transactional DDL means a failure
anywhere in that block leaves the live tables exactly as they were.

**`--into-shadow` requires a complete scope**, and that is a guardrail rather than a tidiness rule.
A swap *replaces* the derived tables; a workspace built from a partial scope is a complete
replacement for the part replayed and an empty one for everything else. So
`--source notes --into-shadow` would delete every other source's corpus, and
`--stage embed --into-shadow` would replace all of it with nothing, having replayed no events at
all. Both would have reported success. A partial scope now refuses, and the in-place path handles
those cases correctly.

The rebuild lands in the workspace via `search_path` on a dedicated engine. The models carry no
schema, so unqualified `memories` resolves to `memoryos_shadow.memories` while `sources` — absent
there — falls through to `public.sources`. The pipeline needs no changes at all, which is why a
schema beats suffixed table names in one schema: with suffixes, every model and query would need a
parallel definition.

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
| `GET /sources`             | Registered sources, with their memory and chunk counts.     |
| `POST /sources/{id}/sync`  | Enqueue a sync. Returns `202` with the job id.              |
| `GET /memories`            | Current memories, filterable by `source_id`, paginated.     |
| `GET /memories/{id}`       | One memory: content, chunks in ordinal order, versions.     |
| `GET /search`              | Semantic search. `source` repeats for several connectors.   |
| `GET /stats`               | What `memoryos stats` prints, from the same function.       |
| `GET /doctor`              | What `memoryos doctor` prints. On demand — it tokenizes.    |
| `POST /judgements`         | Record a verdict. Re-judging replaces.                      |
| `GET /judgements`          | One row per judged query, with verdict counts.              |
| `GET /judgements/export`   | The golden set, ids re-resolved. M2.0's input.              |

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
make test-unit     # unit tests only, no database required
make test          # everything, including integration tests
make test-slow     # the tests that load the real model
make phase1-check  # the whole of Phase 1, from an empty volume
```

Integration tests are marked `integration` and need Postgres running. `slow` tests load the real
model and are excluded from the default run.

**The suite has its own database.** `clean_database` truncates every table, because truncation is
the only isolation strategy that survives code under test committing its own transactions — and
pointed at the development database that is `pytest` deleting a working corpus, which it did three
times during M2.0a. Compose creates `memos_test`; `tests/conftest.py` sets `MEMOS_ENVIRONMENT=test`,
and `Settings` resolves `database_url` to `test_database_url` under that value, so every consumer
lands on it without being told separately. An existing volume predates the init script, so create
it once by hand:

```bash
docker compose exec postgres psql -U memos -d memos -c "CREATE DATABASE memos_test"
```

CI sets neither variable and is unaffected: its database is disposable, so a second one there would
only be a second thing to migrate.

Three tests are load-bearing out of proportion to their size, and each exists because a green suite
was once wrong:

- `tests/slow/test_acceptance.py` — the four assessment queries against the real model. The only
  thing standing between this pipeline and one that fills the column with plausible garbage.
- `tests/unit/test_replay_rules.py::test_every_table_is_classified_exactly_once` — fails when a new
  table is added without deciding whether it can be rebuilt. That omission is otherwise invisible.
- `tests/integration/test_replay.py::test_versions_and_tombstones_survive_a_rebuild` — the test that
  caught the replay applying events without interleaving the pipeline, a defect no row count showed.

If `/health/ready` reports a null `pgvector_version`, the init script did not run because the
data volume already existed. Reset it:

```bash
docker compose down -v && docker compose up -d
```
