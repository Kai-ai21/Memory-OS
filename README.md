# Memory Intelligence OS

A long-term AI memory system. The goal is durable, queryable memory for AI agents — storing
what was learned, retrieving it by meaning rather than by keyword, and keeping it coherent as
it grows. Postgres 17 with `pgvector` is the storage substrate.

## Status

**Phase 1 complete**, plus M2.0a (the search interface), M2.0 (the evaluation harness),
M2.1 (keyword search), M2.2 (hybrid retrieval), M2.3a (measurement reliability),
M2.3b (ranking signals, measured and switched off), M2.4 (cross-encoder reranking)
and M2.5 (citations and explainability).

Point it at a directory and it walks the tree, hashes every file, stores the bytes, records
artifacts and events, versions memories, parses each artifact into normalized text, splits that
text into chunks sized for the embedding model, embeds them, and answers questions about them by
meaning. Then it can throw all of that away and rebuild it from the log, and prove the result is
identical.

Semantic and lexical retrieval, fused by reciprocal rank and rescored by a cross-encoder —
citations and synthesis are the rest of Phase 2. What it retrieves is measured rather than
assumed: see [Evaluation](#evaluation).

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
uv run memoryos search "how a worker takes a task and holds it" -k 5
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
assessment queries, and costs the two-timestamps question the file where the answer is
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
memoryos search "how a worker takes a task and holds it" -k 5
memoryos search "..." --exact          # sequential scan, to see what the index missed
memoryos search "..." --mode keyword   # the lexical half, see below
memoryos eval-recall --queries 50 --ef-search 40,100,200,400
```

```
GET  /search?q=...&k=10&source=NAME&kind=note&after=...&before=...&mode=vector
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

### Keyword search

`--mode keyword` answers from a Postgres full-text index instead of the embedding. Nothing is
combined yet — two retrievers, measured against each other. Fusion is M2.2, and it is separate
so that this comparison exists first.

`memory_chunks.search_vector` is a **generated** `tsvector` column with a GIN index, not a
trigger and not a pipeline step. Postgres recomputes it on every insert and update, so it
cannot drift from `content`; no application code writes it, and none can. A rechunk that
rewrites `content` therefore cannot leave the lexical index describing text that is no longer
there. It also means the migration backfills the whole corpus by itself — adding this needed no
re-ingest.

Three query choices, each with a plausible wrong alternative:

- **`websearch_to_tsquery`**, not `to_tsquery`, which takes an operator language and raises a
  syntax error on ordinary text — turning a user's stray `&` into a 500. Not `plainto_tsquery`
  either, which never throws but also never understands `"a quoted phrase"` or `-exclusion`.
- **`ts_rank_cd`**, not `ts_rank`. Cover density accounts for how close the matched terms sit,
  which is the difference between a chunk containing `SKIP LOCKED` as a phrase and one that
  mentions skipping in a docstring and locking forty lines later.
- **The same eligibility clauses as the vector store**, imported from `db/filters.py` rather
  than retyped. The two retrievers may disagree about ranking; they must agree exactly about
  which rows exist, or M2.2 will fuse a disagreement and it will look like a ranking artefact.

A query that reduces to no lexemes — `"the and of"` — returns an empty list. Finding nothing is
an answer, not an error.

### Hybrid, and why RRF

`--mode hybrid` is the default. Both retrievers run concurrently under `asyncio.gather` — they
are independent queries and serialising them would add the embedding latency to the keyword
latency for nothing — and their two rankings are fused by reciprocal rank.

The obvious alternative, `0.7 * cosine + 0.3 * ts_rank_cd`, does not work and no amount of
tuning fixes it. The two numbers are not on comparable scales: cosine from this model occupies
a narrow band where almost all the range carries no signal, and `ts_rank_cd` is unbounded and
depends on term frequencies across the whole corpus, so the same document scores differently
after ingesting unrelated files. Every normalisation that would make them comparable encodes an
assumption about the score distribution that stops holding as the corpus grows, and it fails
silently — the ranking degrades and nothing errors.

RRF discards the scores and keeps the ordering, which is the part both retrievers mean the same
thing by:

```
score(d) = Σ  weight / (k + rank(d))        k = 60
```

`k = 60` is from the original paper and its job is to flatten the curve, so that agreement
outweighs enthusiasm: a document ranked third by both retrievers scores 1/63 + 1/63 and beats
one ranked first by only one at 1/61. **Not tuned, and deliberately not tuned before M2.3** —
with two retrievers and 21 golden queries there is not enough signal to tell a real improvement
from a fit to this corpus.

Every returned chunk carries a `ScoreBreakdown`: the fused score, and the rank and raw score
from each retriever that found it. This is not a debugging convenience. Once two retrievers
collapse into one number, that number is the only thing anybody sees, and without the
breakdown a fourth-place result cannot be explained, a regression cannot be attributed to a
half, and M2.5's citations have nothing to show.

**Depth is load-bearing.** Each leg fetches `max(k * 3, k * 5)` chunks, the second term being
the same fanout a single-retriever search uses because chunks still collapse into memories at
about 5:1. Taking only the fusion figure left the vector leg shallower inside hybrid than
outside it, and it cost real recall — a pinned answer chunk sat in the band between the two
depths and simply was not in the list to be fused. A hybrid handicapped against its own vector
half measures fanout, not fusion.

**What it bought, and what it cost.** Hybrid beats vector on recall, MRR and nDCG, and sits a
hair below it on precision; it beats keyword on all four. It is never worse than *both*
retrievers on any query — the RRF floor holds — but it is also never better than both, and on
six queries it lands below whichever single retriever was right. The mechanism is worth
understanding before M2.3: when one retriever is uninformative but not *empty*, its rank-1
result still receives the full 1/61, which is enough to displace the other retriever's rank-2.
Equal weights assume both retrievers are equally trustworthy for every query, and they are not.
That is the argument for the weights parameter that already exists on the fusion function and
for whatever M2.4's reranker does with it.

**What the measurement said.** Over the same 21-query golden set, vector wins roughly half the
queries outright, keyword wins four, and the rest tie. The summary undersells it: the query
that named this milestone's motivation went from recall 0.000 on vector to recall 1.000 on
keyword — the one question the semantic half could not answer at all, the lexical half answers
completely — and the union of what the two modes retrieve sits about six points of recall above
vector alone. That headroom is what M2.2 exists to collect. Had the two won on the same
queries, fusion would buy nothing and the milestone would be worth cancelling.

The failures are as complementary as the wins: every query in the set whose answer is written
in words the question never uses scores near-perfectly on vector and exactly **0.000** on
keyword, because not one term in the question appears in the answer.

### Reranking

`--mode hybrid` retrieves a shortlist; a cross-encoder then rescores it. The embedder is
a **bi-encoder** — query and document encoded separately, which is what lets every
document vector be computed once at ingest and searched with a vector lookup, and
exactly what limits it: the model compresses a document into 384 numbers without
knowing what will be asked of it. A **cross-encoder** reads the pair together and scores
it directly. Far more accurate, and impossible to precompute — one forward pass per
query-document pair, at query time. So: retrieve cheaply, rescore a shortlist
expensively, return ten.

**It cannot recover what retrieval missed.** A document outside the shortlist cannot be
ranked into it, which is why this runs after fusion rather than instead of it.

The pair is truncated to the model's reported window before it is scored, and that is
the M1.6.1 lesson applied to a second model. The failure available here is worse than
the original: the pair is `[CLS] query [SEP] document [SEP]`, so an over-long document
does not merely lose its tail — it can push the *query* out of the window and leave the
model scoring a document against nothing.

**The shortlist is 25, and that is measured rather than assumed.** The obvious choice is
50, matching the fanout. Measured over the 41-query golden set:

| shortlist | nDCG@10 | MRR | recall@10 | p95 total |
|---|---|---|---|---|
| off | 0.718 | 0.750 | 0.831 | 39ms |
| 15 | 0.781 | 0.858 | 0.838 | — |
| **25** | **0.788** | 0.829 | 0.839 | **280ms** |
| 50 | 0.761 | 0.802 | 0.820 | 473ms |

A deeper shortlist is both slower *and* worse. It lets the model promote a chunk fusion
ranked fortieth into the top ten, and at that depth its judgement is not better than
fusion's — bounding the shortlist bounds how far a candidate can jump.

**Reranking reorders and adds nothing, and that is checked rather than asserted.** At
k=50 — wide enough that nothing is truncated — the retrieved sets are identical for all
41 queries with and without reranking, and recall@50 and precision@50 match to four
decimal places while MRR and nDCG move. recall@**10** does shift slightly, and that is
arithmetic rather than a bug: reordering changes which ten memories fall above the
cutoff, so a relevant memory can cross it in either direction.

### Citations and explanations

Every result carries the spans it quotes and a reconstruction of why it ranked where
it did. `GET /search` returns both; `?explain=false` omits them, which is the only
thing that skips reading the parent memory's normalized text. `memoryos search
--explain` prints the same thing.

A citation names `(memory, chunk ordinal, char_start, char_end, version)` and quotes
the span. The version is on it because a citation to a memory that has since changed
has to say what it referred to. The excerpt is widened to surrounding context — a bare
chunk is often uninterpretable, since "the second approach" means nothing without the
first — with the boundaries snapped to sentence or line breaks and the span's offsets
*within the excerpt* included, so a client highlights without redoing the arithmetic.

The explanation reconstructs the fusion: each ranking's `weight / (k + rank)` term, and
that term as a **share** of the fused score. The share is the number that answers "why
is this third?", and because it is recomputed from the same arithmetic that produced
the ranking, a test can require the shares to sum to 1.0 — an explanation that has
drifted from the ranker is worse than none. The one-sentence `why` is assembled from
those numbers, never generated: it has to be available on every result, cost nothing,
and say the same thing twice.

**`memoryos verify-citations` is the milestone's real deliverable.** It asserts one
identity on real rows:

```
memory.content[char_start:char_end] == chunk.content[prefix_chars:]
```

M1.4a broke exactly that, and nothing noticed — row counts were right, offsets were in
bounds, every test passed, and highlights pointed a few hundred characters from the
answer. This is the standing check that would have caught it on day one. It exits
non-zero on any mismatch, and a test corrupts a chunk on purpose to prove it can fail,
because a verification that cannot fail proves nothing.

### Ranking signals, and why they are off

M2.3b added recency and importance as two more rankings into the same RRF, with
`MEMOS_WEIGHT_RECENCY` and `MEMOS_WEIGHT_IMPORTANCE` controlling how much they count,
and `memoryos tune-weights` searching that space against the golden set. Both default
to **zero**, and that is the result rather than an unfinished feature.

Recency decays exponentially with a 180-day half-life, and an undated item scores 0.5
rather than 0 — an unknown date is not evidence of age, the same principle as the
constraint that forbids substituting `ingested_at` for a missing `occurred_at`.
Importance is a labelled proxy over three observable things: chunk count log-scaled so
it does not become a proxy for file size, revision count, and edit freshness. No model
assigns it; `memoryos recompute-importance` computes it, off the ingest path because
two of its three inputs are properties of an item's history rather than of the bytes
just read.

Both enter fusion as *rankings over the candidates the retrievers already found*, never
as multipliers on a fused score and never as a source of new candidates. A file can be
promoted for being recent; it cannot appear for it.

Then the grid was searched, 97 combinations across a coarse and a fine pass, and it
said no:

- **Recency monotonically lowers nDCG.** 0.735 at weight 0, 0.731 at 0.15, 0.721 at
  0.30, 0.707 at 0.60 — at every importance level tried. On a repository of
  explanatory prose, when a file was last edited says almost nothing about whether it
  answers a question about the design.
- **The best importance weight gains 0.0109**, which is below the 0.0122 resolution
  floor. Not evidence. Shipping it would be shipping noise with a decimal point.

There is one real effect worth recording: recency trades recall for MRR. At weight 0.6
it lifts MRR from 0.762 to 0.839 while dropping recall from 0.852 to 0.779 — it pushes
one right answer to the top and several others out of the top ten. For a system whose
next milestones are citations and synthesis, that is the wrong trade.

The tuning had no train/test split. 41 queries cannot spare a held-out set without
every score moving further than the effect being measured, so the winner is compared
against the floor rather than against the runner-up, and `tune-weights` prints that
judgement itself instead of leaving it to the reader.

**A hygiene rule this section had to learn.** Do not write golden-set queries verbatim into the
corpus. This repository is what gets indexed, so a paragraph quoting a query string makes that
paragraph a literal match for it — and the lexical retriever is built to find literal matches.
An earlier draft of this section quoted three queries and moved two of them from 0.000 to
1.000, measuring the README rather than the retriever. Describe the queries; do not quote them.
`SKIP LOCKED` is the exception that cannot be avoided, because it is a real clause this project
uses and documented long before it became a test case.

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
"right file, wrong chunk": M2.0 found the two-timestamps question returning both
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

The baseline is 41 queries over 440 judgements at k=10, and it lives in
`var/baseline.json` — the four means are deliberately **not** copied into this file.

That is not laziness, it is the corpus. This repository is what gets indexed, so a
paragraph quoting the score used to be part of what produced the score. M2.3a fixed
that (see below) and the numbers are now stable under documentation edits, but the
habit stays: a measurement belongs in the artefact that recorded it, and
`evaluate --compare` against the committed JSON is the interface.

### Measurement reliability

M2.2 could not tell a real improvement from noise. Rewriting one README section moved
mean MRR by 0.009 — larger than the difference being reported — and reversed the sign
of the vector/hybrid gap. Two causes, both now fixed.

**Self-reference.** Some files exist *because of* the golden set: the acceptance test
runs the assessment queries verbatim, and `phase1-check` demonstrates them as shell
commands. The lexical retriever ranked those files first for exactly those queries,
finding the test that names the question rather than anything answering it.
`eval_exclude` in the golden set file drops matching memories from every ranking
*before* any metric sees them, over-fetching so a filtered run still scores k results,
and the harness reports what it dropped per query. The README is deliberately not
excluded — it holds real answers, and `tests/unit/test_golden_hygiene.py` keeps it
honest instead, failing when any natural-language golden query appears verbatim in a
tracked file. Code literals are exempt: a repository that did not contain
`SKIP LOCKED` would make that query meaningless.

**Sample size.** 21 queries made one query 5% of the mean. There are now 41, spread
across the four things the earlier milestones showed matter: explanatory prose whose
answer shares no words with the question, literals that appear only inside code,
instructional questions whose answer repeats the question, and questions no single file
answers.

**The resolution floor**, measured rather than assumed, with `evaluate --repeat` and by
perturbing the corpus:

| probe | movement |
|---|---|
| same evaluation run three times | 0.0000 |
| an unrelated file edited and re-ingested | 0.0000 |
| a file inside the answer space edited | 0.0000 |
| this README edited with neutral prose — the M2.2 failure, repeated | 0.0000 |
| writing *this section*, which is about retrieval and several pages long | 0.0011 (nDCG only) |
| *control:* a new file added that genuinely answers a query | 0.0122 |

The control matters: without it, the zeroes would mean the probe was broken rather than
the harness stable. Retrieval is deterministic, so a difference measured over an
unchanged corpus is real at any size; across corpus states, one file entering the
answer space moves MRR by about 0.012, and that is the margin a cross-corpus claim
needs.

The second-to-last row is the honest version of the first: a neutral paragraph moves
nothing, and several pages *about retrieval* still move one metric by 0.0011 — an
eighth of what the same operation cost M2.2, and now smaller than anything worth
reporting.

**The worst-queries section is the useful output.** The mean says whether something
improved; the worst list says what to fix. The bottom of it is stable across runs:
the rare-literal query, the deduplication question, and the one about running
worker`. The first two are one defect approached from opposite sides — a query whose
answer is a rare literal token the model reads as ordinary English, and a query whose
answer is phrased entirely in words the code never uses. Between them they are the case
for M2.1's lexical half.

The second failure the chunk ordinal exposes: the two-timestamps question takes a
perfect MRR and loses two fifths of its recall, because `README.md` and `models.py` are
both inside the top six on paragraphs that never mention `ingested_at`. Judged per
memory that query looks solved.

`var/baseline.json` is committed. It is the record of where Phase 2 started, and every
later milestone in this phase reports its `--compare` diff against it. `var/baseline-hybrid.json`
sits alongside it rather than replacing it: two retrieval systems now exist, and the vector
number is the fixed point the phase is measured from. A run records which mode produced it, and
`--compare` says so loudly when the two disagree, because comparing a keyword run against a
vector baseline otherwise reads as a catastrophic regression.

**A contamination this milestone surfaced.** `tests/slow/test_acceptance.py` contains several
golden queries verbatim, because it is the acceptance test for them — and the corpus is this
repository. The lexical retriever therefore ranks that file first for those queries, finding
the test that names the question rather than anything that answers it, and RRF then promotes
it. It is the same hazard the keyword section warns about, arriving through a file that has a
legitimate reason to hold the strings. Worth fixing before the golden set grows further.

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
