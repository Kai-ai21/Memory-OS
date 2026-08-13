# Memory Intelligence OS

A long-term AI memory system. The goal is durable, queryable memory for AI agents — storing
what was learned, retrieving it by meaning rather than by keyword, and keeping it coherent as
it grows. Postgres 17 with `pgvector` is the storage substrate.

## Status

**Phase 1 complete**, plus M2.0a (the search interface), M2.0 (the evaluation harness),
M2.1 (keyword search), M2.2 (hybrid retrieval), M2.3a (measurement reliability),
M2.3b (ranking signals, measured and switched off), M2.4 (cross-encoder reranking),
M2.5 (citations and explainability) and M2.6 (grounded answers). **Phase 2 complete.**
**Phase 3 complete**: M3.0 (Neo4j and the graph schema), M3.1 (entity extraction),
M3.2 (entity resolution), M3.3 (typed relationships), M3.4 (projection sync, rebuild and
divergence detection) and M3.5 (graph-augmented retrieval, measured and shipped at weight
zero). See [Graph](#graph) and [Graph-augmented retrieval](#graph-augmented-retrieval).
**Phase 4 complete**: M4.0 (the temporal query layer), M4.1 (the timeline view),
M4.2 (evolution and change detection) and M4.3 (time-aware retrieval, measured).
See [Time](#time) and the [Phase 4 retrospective](#phase-4-retrospective).
**Phase 5 begun**: M5.0 (the decision schema and capture) — what was decided,
what else was considered, why, and what had to be true — M5.1 (outcome linking),
which connects a decision to what happened afterwards and is the milestone where
Phase 4's temporal layer pays for itself, M5.2 (assumption tracking), which
evaluates which of those beliefs held, and M5.3 (pattern discovery), which finds
none and explains exactly why. See [Decisions](#decisions), which opens
with the measurement of how little decision-shaped content this corpus actually
holds, [Outcomes](#outcomes), [Assumptions](#assumptions) and [Patterns](#patterns).

Point it at a directory and it walks the tree, hashes every file, stores the bytes, records
artifacts and events, versions memories, parses each artifact into normalized text, splits that
text into chunks sized for the embedding model, embeds them, and answers questions about them by
meaning. Then it can throw all of that away and rebuild it from the log, and prove the result is
identical.

Semantic and lexical retrieval, fused by reciprocal rank, rescored by a cross-encoder, and
answered in prose that cites its sources or declines. What it retrieves is measured rather
than assumed: see [Evaluation](#evaluation).

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

Seven tables at M1.1, split into two groups that matter more than the count.
`ingestion_events`, `raw_artifacts` and `sources` are the source of truth and are never
truncated. `memories`, `memory_chunks`, `jobs` and `embedding_cache` are derived, and M1.7
rebuilds them from the first group plus the blob store. The split is declared as data in
[`application/replay.py`](src/memoryos/application/replay.py), and a test fails if a new table is
not classified.

Later phases added a third group rather than widening either of these. `query_judgements`
(M2.0a), the five decision tables (M5.0), the three outcome tables (M5.1) and the
three assumption tables (M5.2) are **user-authored**: neither rebuildable nor ingestion input, never truncated and never
written by a replay. Three of them — `decision_evidence`, `outcome_evidence` and `assumption_evidence`
— hold cascading foreign keys into `memories` anyway, so a replay snapshots their links by natural
key and re-links them afterwards; a test derives that list from the metadata, so a fourth
such table fails the build rather than losing its rows in a cascade nobody watched. See
[Replay](#replay), [Decisions](#decisions) and [Outcomes](#outcomes).

| Table              | Holds                                                              |
| ------------------ | ------------------------------------------------------------------ |
| `sources`          | Where artifacts come from, plus opaque per-source sync cursors.     |
| `raw_artifacts`    | Content-addressed artifacts. The BLAKE2b-256 hash is the key.       |
| `ingestion_events` | Append-only log of everything observed. Replayed to rebuild state.  |
| `memories`         | One row per version of an item; one current version per item.       |
| `memory_chunks`    | Retrievable spans with offsets and a 384-dimension embedding slot.  |
| `jobs`             | Durable work queue. Derived: a rebuild empties it.                  |
| `embedding_cache`  | Vectors keyed by (model, role, text). Content-addressed memoisation. |
| `query_judgements` | M2.0a. A human's verdict on one result for one query. User-authored. |
| `decisions`        | M5.0. What was decided, why, and how sure. User-authored.           |
| `decision_options` | What else was on the table, and why each alternative lost.          |
| `decision_assumptions` | What had to be true. `held` is written by M5.2, null until then. |
| `decision_evidence` | Memories that informed, record, or contradict a decision.          |
| `decision_suggestions` | The review queue. A draft is not a decision until accepted.     |
| `decision_outcomes` | M5.1. What happened afterwards, and whether anybody watched it. |
| `outcome_evidence` | Memories showing an outcome happened, with the date snapshotted. |
| `outcome_suggestions` | Candidates, with the temporal gap and shared entities kept.   |
| `assumption_groups` | M5.2. Assumptions from different decisions saying one thing.    |
| `assumption_group_candidates` | Pairs the embedder was unsure about.                  |
| `assumption_evidence` | Memories bearing on whether an assumption held.              |
| `patterns`         | M5.3. A behavioural claim, with the evidence that makes it one. |
| `pattern_evidence` | Decisions that support a pattern, and decisions that contradict it. |

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

### Grounded answers

```bash
memoryos ask "how does the job queue prevent two workers claiming the same job"
memoryos eval-answers --json var/answers.json
```

```
POST /answer   { "q": "...", "k": 10 }
```

This is the only part of the system that can *invent* something. Everything else returns
text it retrieved — wrong, badly ranked or stale, but never fabricated. A model asked to
answer from context will smooth over a gap with a fluent claim from its training data,
and nothing in the prose marks it as different from a grounded one.

So the guardrail is mechanical rather than prompted. The prompt asks for citations and
for refusal when the passages fall short; `domain/grounding.py` then checks what came
back, and the response carries that check rather than a promise:

- **Citation indices outside the supplied range** are the unambiguous failure — the
  model referenced a passage that was never in the prompt. Reported, and they resolve to
  no citation, because there is nothing behind them.
- **Factual sentences with no citation** are flagged, never removed. Quietly deleting a
  sentence mid-answer produces prose that reads complete while missing a step; a marker
  says which part is not grounded.
- **A refusal scores 1.0**, not 0. It contains no claims to cite, and scoring it zero
  would make the safest answer look like the worst one.

Context assembly counts tokens with the model's own tokenizer — M1.6.1's lesson — and
**drops whole passages rather than truncating one**. A passage cut mid-sentence is worse
than an absent one: the model cannot tell the sentence was severed, completes the thought
from training data, and produces a fabricated claim carrying a citation to a real
passage. That is the most convincing wrong answer this system can make.

Passages go *before* the question in the user message, because long-context models attend
better to material preceding the instruction, and the instruction — refuse if the
passages fall short — is the part that must not be forgotten.

`var/refusal-queries.json` holds ten questions this corpus cannot answer. An answer to
any of them is a fabrication, and the refusal rate on that set is worth more than any
accuracy metric. `eval-answers` exits non-zero if any is answered rather than declined.

Answering is the only feature that needs a credential (`MEMOS_GEMINI_API_KEY`). Without
one, search, retrieval and every measurement above work unchanged, and `ask` says so.

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

## Graph

M3.0 adds Neo4j alongside Postgres. Infrastructure only at that milestone — the schema, the port
and the adapter — with M3.1 to M3.3 filling it and M3.4 deciding who is allowed to write to it.
See [Projection sync](#projection-sync).

**Why a graph rather than more Postgres tables.** Fixed one- and two-hop relationships are
genuinely fine in SQL and frequently faster; a join table with the right index beats a graph
database at "which entities does this memory mention". The break comes at *variable-depth*
traversal. "What connects this decision to that person?" is two hops or five, and which one is not
known when the query is written. In SQL that is a recursive CTE whose readability and cost both
degrade with depth; in Cypher it is `[*1..5]`. Traversal also costs what the neighbourhood costs
rather than what the tables cost, because relationships are stored as pointers instead of being
resolved through an index on every hop.

**Everything in the graph is a projection.** `Memory` nodes carry `memory_id`, `external_key`,
`kind` and `occurred_at` — no content and no chunks. Postgres stays the system of record, and a
copy of the text here would be a second thing to keep correct and a second answer to give when the
two differ. On any disagreement Postgres wins and the graph is rebuilt, which is why no use case
may write to Neo4j directly: writes that bypassed the rebuild would survive exactly until the next
one.

`GraphStore.clear()` exists to make that concrete, and it is only affordable because Phase 1
designed the derived state to be disposable. `DERIVED_PROJECTIONS` in `application/replay.py`
classifies the graph the way `DERIVED_TABLES` classifies the tables — a separate set, because
everything reading that tuple puts its contents inside a `TRUNCATE`.

**A full replay clears the graph and rebuilds it; a scoped one does neither.** After a
whole-corpus rebuild every memory id is new, so every node the graph held refers to a row that no
longer exists. A scoped replay is the opposite case: `clear()` empties the entire projection, so
calling it to rebuild one source would discard every other source's nodes. `verify-replay` never
touches the graph at all, because it is a read-only comparison — what it compares instead is the
projection each side of the rebuild *implies*, read from Postgres. See
[Projection sync](#projection-sync).

**There is no Alembic for Neo4j.** The constraints and indexes live in `adapters/graph/schema.py`,
every statement carries `IF NOT EXISTS`, and the whole set is applied on first use rather than by a
migration somebody has to remember to run. A `:SchemaVersion` node records which revision was
applied, so a database that predates a constraint is distinguishable from one that has it —
`memoryos doctor` reports the drift. Uniqueness on each label's identity property is not only a
constraint but the index that backs `MERGE`: an unconstrained merge scans, and under concurrency
two transactions can both find nothing and both create.

**An unreachable Neo4j is degraded, not failed.** `/health/ready` returns 200 with
`status: degraded` and `graph: false`. A 503 would be the wrong kind of correct — the body would
accurately say the graph is down, and an orchestrator reading the code would remove an instance
that can still serve every Phase 1 and Phase 2 request. The status code answers "should traffic
come here?"; the body answers "what works?".

```bash
docker compose up -d          # Postgres on 5433, Neo4j on 7474 (browser) and 7687 (bolt)
memoryos doctor               # reachability, schema version, node counts by label
```

Graph tests are marked `graph` and skip when Neo4j is absent, so the suite still runs for
everything Phase 1 and Phase 2 do. CI starts a Neo4j service, because a permanent skip is
indistinguishable from a passing test.

Isolation works differently here than everywhere else in the suite, and it has to. Postgres gives
the tests their own database to truncate; Neo4j Community Edition supports exactly one user
database, so a test run and a developer's graph share it. Tests therefore isolate by identity —
every id comes from `GraphFixture.new_id`, which records it, and teardown deletes those nodes and
nothing else.

Which leaves the assertions that are about the *whole* graph — a rebuild, a clear, a divergence
check — with nowhere to run. Those use `InMemoryGraphStore`, a `GraphStore` with `MERGE` semantics
and no database, because what they test is arithmetic over two projections and a database does not
make it more true. The one claim a fake cannot make is that Cypher reads back exactly what it was
given, so that is asserted against a real Neo4j, scoped to minted ids, in
`test_the_projection_reads_back_exactly_as_written`. Every in-memory assertion rests on it.

## Entity extraction

M3.1 extracts entities from every chunk with the configured LLM, stores them in
Postgres, and projects them into the graph. No resolution and no relationships —
those are M3.2 and M3.3.

```bash
memoryos extract-entities --limit 20 --dry-run   # what would be extracted
memoryos extract-entities --source notes         # do it
memoryos entity-stats                            # counts, top entities, duplicates
```

**Offsets are verified, never trusted.** The model returns a name; it is not
asked where the name appears, because a language model cannot count characters
and the answer would have to be discarded anyway. The name is located in the
text instead, and one that cannot be found is dropped and counted. That counter
is the only direct measurement of fabrication this system produces.

The check runs twice — once in the adapter, once at the storage boundary — on
the same reasoning as `EmbedMemory._check_widths`. `entity_mentions` exists to
carry `content[char_start:char_end] == name`, and an invariant that holds only
while every present and future extractor keeps its word is not an invariant. One
string slice makes it unconditional.

**Provider-agnostic.** Extraction goes through the `LanguageModel` port, so it
runs on whatever `MEMOS_LLM_PROVIDER` selects. A provider-specific extractor
would have rebuilt the coupling M2.6 removed.

**Batched, because the free tier's binding constraint is requests per day.**
1,308 chunks one at a time exceeds the cap before it exceeds anything else.
Chunks are numbered in one prompt and demultiplexed by index; a chunk
misattributed by the model produces a name that is not in that chunk's text and
is dropped by the same rule that catches invented ones. `MEMOS_EXTRACTION_BATCH_SIZE`
tunes it.

**Malformed JSON gets one retry with a blunter instruction, then
`PermanentError`.** A model that cannot produce JSON twice will not produce it
on the fifth attempt, and retrying to exhaustion multiplies a broken prompt by
the attempt budget against a per-day quota.

**Idempotent on `extractor_version`**, which encodes the prompt version and the
model id. A memory whose mentions already carry the current version is skipped
without a model call, so re-running the command is free and a prompt improvement
is a targeted redo rather than a corpus rebuild.

`entities` carries `UNIQUE (canonical_name, type)`, which the milestone's schema
did not specify and cannot do without: every re-extraction would otherwise insert
another row for the same name, and "most-mentioned entities" would be a list of
coincidences. `canonical_name` is casefold plus collapsed whitespace and nothing
more — normalising harder would shrink the duplicate count M3.2 is scoped
against, which is improving a number by moving the ruler.

Both tables are classified `DERIVED`, and they stretch the word hardest of
anything in that set. A replay empties them and does not rebuild them: doing so
would mean an LLM call per chunk on every verification run, and every chunk id is
new after a rebuild, so every mention would dangle regardless. Re-run
`extract-entities` after a full replay.

## Entity resolution

M3.2 merges entities that refer to the same thing. This is where a graph phase
succeeds or fails: an unresolved graph looks impressive in a picture and is
useless to traverse, because the path you need runs through a node that exists
four times under four spellings and therefore does not exist at all.

```bash
memoryos resolve-entities --dry-run     # proposals with evidence, changes nothing
memoryos resolve-entities               # apply the certain ones, queue the rest
memoryos entity merges --pending        # the review queue
memoryos entity merge <a> <b>           # your verdict on a pending pair
memoryos entity unmerge <merge-id>      # undo any merge
```

**One asymmetry decides every threshold here: a false merge is worse than a
missed one.** A missed merge leaves a traversal short a path — visible, and
fixed by a later merge. A false merge *invents* a path, and every traversal
through it reports a connection nobody wrote.

Three strategies, cheapest first:

1. **Exact canonical match**, from `domain/canonicalize.py`. Type-specific,
   because the rules contradict each other: stripping a trailing `js` is right
   for a TECHNOLOGY and destroys a FILE called `index.js`. `sql` is deliberately
   *not* stripped — it turns "PostgreSQL" into "postgre" and "MySQL" into "my",
   forms that match nothing at all.
2. **Embedding similarity**, reusing the corpus's own `Embedder`. It reaches
   pairs no character rule can, and it **proposes rather than decides** — see
   below.
3. **Aliases**, a deliberately tiny hand-written table. A long alias list is a
   resolver somebody is maintaining by hand instead of fixing.

Blocking keeps this out of O(n²): only entities of the same type are compared.

**Embedding similarity does not auto-merge, and that is measured rather than
cautious.** The milestone proposed auto-merging above a high threshold. On this
corpus no such threshold exists — two different constraint names scored 0.952
while a real match (`ingestion_events`/`IngestionEvent`) scored 0.939. A
bi-encoder embeds identifier-like names by their shared prefix, and a shared
prefix is not a shared referent. So embedding feeds the review queue and a
person decides.

**Nothing is deleted.** The losing entity keeps its row, gains a
`merged_into_id`, and its mentions are repointed at the winner. The ids that
moved are recorded on the merge, because after a repoint nothing distinguishes
the mentions that came from the loser from the ones the winner always had — that
record is the difference between `unmerge` restoring the previous state and
restoring something that resembles it.

Extraction follows `merged_into_id`. Without that, the next run re-attaches
mentions to a merged-away row by canonical name and silently undoes the
resolution.

**Known gap:** `entity_merges` is classified derived, forced by its foreign key
to `entities`, which a replay truncates. Automatic merges are rebuildable by
re-running the resolver; *manual* merges and the pending queue are not, and a
full replay loses them. The fix is M1.7's — key merges on `(canonical_name,
type)`, which survives a rebuild, rather than on entity ids, which do not.

## Relationships

M3.3 adds typed, directed edges between resolved entities, each carrying the
chunk that asserted it.

```bash
memoryos extract-relationships --limit 20 --dry-run
memoryos extract-relationships
```

**Typed and directed, because neither is decoration.** An untyped edge says two
things are related, which is close to useless for traversal: at depth three
every entity relates to every other, so following "related" returns the corpus.
`USES` and `AUTHORED_BY` let a question be asked. And direction cannot be
flattened — "A supersedes B" and "B supersedes A" are contradictory claims about
the same pair.

Predicates: `USES`, `DEPENDS_ON`, `PART_OF`, `AUTHORED_BY`, `MENTIONS`,
`SUPERSEDES`, `RELATES_TO`. Closed, for the reason `EntityType` is: an open
vocabulary gives you `USES` beside `USED_BY` beside `UTILIZES`, which is three
names for one traversal that each return a third of it.

**Every row carries its chunk.** Without provenance a relationship is an
unfalsifiable claim, and a Phase 3 answer is only as citable as its
least-supported edge — the property M2.5 established for Phase 2 answers.

**The same claim in five chunks is five rows.** That is evidence, not
duplication: M3.5 weights edges by assertion count, so a claim and a
five-times-repeated claim have to stay distinguishable. The unique constraint is
scoped per chunk to keep them so, and the graph collapses them into one edge
carrying `assertion_count` — connectivity and evidence being different questions.

**The model links between numbers, never names.** It is given the chunk's
already-resolved entities as a numbered list and returns indices, so there is no
name to hallucinate; an out-of-range index is dropped and counted. An edge to a
fabricated entity looks exactly like a real one until somebody follows it.

Entities are the *resolved* ones from M3.2, and the ordering matters: extracting
before resolution would attach edges to entities about to be merged away, each
needing re-pointing by hand.

In Neo4j every predicate becomes a `RELATES_TO` edge carrying `predicate`,
`confidence` and `assertion_count` as properties, rather than seven new
relationship types. M3.0's traversals are written against three types, and
promoting seven more would force every existing pattern to enumerate them for no
gain — `[:RELATES_TO {predicate: 'uses'}]` filters just as precisely.

Only chunks naming two or more entities are sent to the model. A chunk with one
entity has nothing to relate, and on a rate-limited tier a request spent being
told so is the difference between finishing a corpus and stopping partway.

## Projection sync

Postgres is the system of record; Neo4j is a projection. M3.4 makes that
operational rather than aspirational: **nothing writes to Neo4j from a use case.**

```bash
memoryos graph rebuild --dry-run   # what would be projected, touching nothing
memoryos graph rebuild             # clear and re-project everything from Postgres
memoryos graph verify              # compare the two; exits non-zero on divergence
memoryos graph sync --memory <id>  # re-project one neighbourhood, inline
```

Ingestion, extraction and resolution enqueue a `SYNC_GRAPH` job naming what
changed; `application/graph_sync.py` is the only writer. M3.1 and M3.3 projected
inline, which is a use case writing to two stores — the graph becomes a second
source of truth in everything but name, however carefully the write is ordered.

The queued version also fixes the failure handling. An inline projection had to
catch its own errors and report a degraded graph, because the rows were already
committed and taking the job down would turn a projection outage into a failed
extraction. A queued sync has no such conflict: it raises, the worker retries it
with backoff, and a Neo4j that was down for a minute is a pause rather than a
divergence nobody records.

**One definition of the projection.** `application/graph_projection.py` reads
Postgres into data; `write` puts that data into a store, `content_hash` hashes it,
and `graph verify` compares the two. A projector that wrote as it read could only
be verified against itself. Three writers with three ideas of what the graph
should hold is what M3.3 shipped, and two of them were wrong — see the commit
that removed them.

```
(:Source)<-[:FROM_SOURCE]-(:Memory)-[:MENTIONS]->(:Entity)-[:RELATES_TO]->(:Entity)
```

Every current, undeleted memory is projected, not only the ones extraction has
reached. The old rule made the graph's memory count a fact about extraction
progress and left a divergence check unable to tell a missing node from an
unextracted one. That is also why ingestion enqueues a sync: without it the only
thing announcing a memory to the graph was extraction, so a deployment with no API
key projected nothing and `graph verify` reported permanent divergence.

### Delete, then project. Never patch.

A scoped sync is a rebuild of a neighbourhood and works exactly as the full
rebuild does: prune the scope, then write what Postgres says that scope should be.
An additive sync converges on the graph *containing* everything Postgres implies
and never removes what it has stopped implying — a mention dropped by a
re-extraction, a merged-away entity, an edge whose row is gone. None of those is
visible in a node count.

That makes idempotence structural rather than tested for. What the test actually
checks is the **scope expansion**, which is the only difficult part. Pruning is
detaching, so removing an entity takes every `MENTIONS` edge into it, including
edges from memories the payload never named. So a payload is widened until it is
closed under one step of the mention relation, against *both* stores:

- a memory in scope brings the entities it mentions;
- an entity in scope brings every memory that mentions it — according to Postgres,
  which knows what is true now, **and** according to the graph, which knows what
  it currently claims.

The graph's opinion is not redundant, and two cases show why. An entity that has
just lost its last mention is unreachable from Postgres, but its node is still
there with an edge from a memory in scope; without the graph's answer it survives
every scoped sync as an orphan. A merged-away loser is the mirror image.

Expansion through a hub is expensive and visible: on this corpus, syncing the one
memory that mentions `sqlalchemy` and `postgres` widens to 18 memories and 185
entities. Above 200 memories the sync escalates to a full rebuild and says so,
because pruning most of the graph a node at a time is strictly more work than one
`DETACH DELETE`.

### There is no shadow swap, and it is not for want of trying

The Postgres equivalent builds the replacement in a second schema and moves the
tables in one transaction, so the corpus is never unavailable. Neo4j Community
Edition offers nowhere to build a replacement — measured on 5.26.29:

```
CREATE DATABASE memosshadow            -> Unsupported administration command
CREATE ALIAS memosalias FOR DATABASE   -> Unsupported administration command
CREATE COMPOSITE DATABASE memoscomposite -> Unsupported administration command
```

Refused by *edition*, not by permissions: `SHOW DATABASES` returns exactly
`neo4j` and `system`. Nor can it be faked inside the one database with a parallel
set of labels, because Cypher has no operation that renames a label or a
relationship type — the "swap" would be a write over every node and edge, which is
the rebuild again with an extra copy of the data.

**So the rebuild accepts documented downtime**, and reports it: the graph is empty
between the clear and the last write, which is 1.7s for 450 nodes and 577 edges on
this corpus. It grows with the corpus, which is the honest argument for the
incremental sync existing at all — a rebuild after every change would work today
and would not at ten times the size.

### Divergence detection

`graph verify` compares counts *and* a content hash of every node and edge, per
type, and exits non-zero on any difference. The rules differ from
`verify-replay`'s in one way worth stating, because it looks like an
inconsistency: this compares on **primary keys**, where the replay check compares
on natural keys. A rebuild mints new ids, so an id comparison there fails on every
honest replay; here the graph *copies* Postgres' ids by construction, so an
`Entity` node whose `entity_id` names no entity is a real divergence.

Content and not counts, for the reason M1.6.1 established: counts match while a
node carries a stale name. Proven by corrupting one — measured on the live corpus,
`SET e.name = 'SQLAlchemy CORRUPTED'` on two nodes:

```
[ok  ] Memory               162       162   79cf00e96740
[FAIL] Entity               287       287   c5b65b1ef6c2 != 3f25b5167011
         differs: 019ff00d-... (SQLAlchemy CORRUPTED (technology)): ('sqlalchemy', ...) -> ('SQLAlchemy CORRUPTED', ...)
```

Both counts unchanged, exit code 1. Nothing is repaired: `graph rebuild` is the
repair, and keeping them separate leaves somebody able to answer how often the
projection actually diverges.

A node carrying a label the projection does not define is counted separately from
a stale one, because it is a different kind of wrong — a stale `Entity` is a
projection that is behind, a foreign label is a writer nobody knows about.

### The predicate is part of an edge's identity

Found by `graph rebuild --dry-run` reporting 25 `RELATES_TO` where the graph held
24. Neo4j merges one relationship per `(type, start, end)`, so "sqlalchemy uses
postgres" and "sqlalchemy depends_on postgres" — both real claims — collapsed into
a single edge whose predicate was whichever the projection wrote last, and which
one survived depended on row order. Nothing failed.

`GraphEdge.identity` names the properties that belong inside the `MERGE` pattern
rather than in the `SET` after it. `domain.values.EDGE_IDENTITY_PROPERTIES` is the
closed set they may come from, because a property name is interpolated into Cypher
for the same reason a label is: it cannot be a bound parameter.

### A replay clears the graph and rebuilds it

M3.0 cleared and stopped, and empty is not the same as correct — a projection
nobody has built is indistinguishable, to `graph verify`, from one that has
diverged. A full replay now clears before the tables are rebuilt (so the graph
never holds ids Postgres has deleted) and projects after.

What it projects is **not what was there before**, and that is the honest outcome
rather than a gap. `entities`, `entity_mentions` and `entity_relationships` are
derived tables a replay truncates and deliberately does not rebuild — an LLM call
per chunk, against offsets that no longer point anywhere — so the projection after
a full replay holds every memory and its source and no entities at all. Re-run
`extract-entities`, `resolve-entities` and `extract-relationships` afterwards.

`verify-replay`'s snapshot gains `graph_memory_nodes`, compared on natural keys
like every other row in it. The entity half is *reported* and not diffed, for
exactly the reason `Snapshot` has never held an entity row: a diff of a section
that is empty by design on one side is a check that fails every time and teaches
nobody anything.

## Graph-augmented retrieval

M3.5 adds a fifth ranking to M2.3's weighted RRF:

```
rankings = [vector, keyword, recency, importance, graph]
```

```bash
MEMOS_WEIGHT_GRAPH=1.0 memoryos search "what shares an entity with the parser registry" --explain
memoryos evaluate --k 10 --compare var/baseline-hybrid.json
memoryos tune-weights --grid coarse
```

**What it is for.** Vector retrieval finds text that means the same thing; keyword
retrieval finds text that says the same thing. Both are similarity relations over
one document at a time, and both are blind to the same class of answer: the memory
that shares no vocabulary and no paraphrase with the query and is relevant because
it is *about the same things*.

**Why it might not work.** An entity mentioned in fifty memories connects all
fifty, so expanding along it produces fifty weakly-related candidates and calls
them evidence. Two guards decide which happens.

### Hub suppression, which is IDF applied to graph nodes

An entity appearing in more than `MEMOS_GRAPH_HUB_RATIO` of the reachable memories
(default 10%) is excluded from the traversal, and everything below the threshold is
weighted by `log(1 + N/df)`. So a shared mention of `SKIP LOCKED` counts for far
more than a shared mention of `postgres` on a corpus about Postgres.

Two details earn their place. The exclusion is applied to **every node a path
crosses**, not to its endpoints: a hub excluded only as a destination is still a
bridge, and at depth 2 a bridge connects everything that mentions it to everything
else that does. And the denominator is the number of memories the graph can
*reach* — memories with at least one mention — not the size of the corpus, because
an entity in 20 of 34 extracted memories is a hub of everything the graph knows.

The threshold is floored at 2. Without the floor, `ceil(3 × 0.10) = 1` makes every
entity a hub on a small corpus, expansion reaches nothing, and the graph looks
useless when what was wrong was the arithmetic.

### Depth, in entity hops

`MEMOS_GRAPH_DEPTH` counts *entity* hops and defaults to 2; the Cypher bound is
twice that, because an entity reaches another either directly through a
`RELATES_TO` edge or through a memory that mentions both. Depth 3 on a graph this
connected reaches most of the corpus, and a ranking that contains everything is
not a ranking.

### The one ranking that introduces candidates

M2.3's recency and importance signals only reorder what the retrievers found — a
document can be promoted for being recent but cannot appear for it. Graph
expansion has to introduce rows, because the memory that shares no vocabulary with
the query is by construction not in either retriever's list. That is the point, and
it is also the risk, and it is why `MEMOS_WEIGHT_GRAPH` defaults to **0.5** rather
than 1.0: a graph-only candidate contributes 0.5/61 against a retriever's 1/61, so
a chunk both retrievers rank first still outranks anything the graph found alone.

At weight 0 the expansion does not run and its candidates never reach the fusion —
not merely a zero-weighted term, because a chunk fused at score zero would sort
below everything and change the result. That is what makes "graph weight 0
reproduces M2.3 exactly" true by construction.

`ScoreBreakdown` gains `graph_rank`, `graph_score` and `graph_path`, and the path
is not a debugging aid. Expansion is the ranking whose contribution a reader is
least able to reconstruct — a promoted result may share no word with the query — so
`search --explain` prints the route:

```
from: keyword #3 (0.0891)  graph #2 via job queue -> SKIP LOCKED
```

### What it measured, and why the weight ships at zero

The milestone specified `MEMOS_WEIGHT_GRAPH=0.5`. It ships at **0.0**, because two
measurements overrode it. Both are controlled A/Bs — same code, same corpus, same
46-query golden set, graph off versus on — because the committed
`var/baseline-hybrid.json` predates M2.4's reranking and comparing against it would
mix two changes.

| configuration | queries the graph reached | results it alone found | recall@10 | precision@10 | MRR | nDCG@10 |
| ------------- | ------------------------- | ---------------------- | --------- | ------------ | --- | ------- |
| graph off (control) | — | — | 0.809 | 0.472 | 0.814 | 0.768 |
| weight 0.5, hubs 10% | 0/46 | 0 | +0.000 | +0.000 | +0.000 | +0.000 |
| weight 1.0, hubs 10% | 9/46 | 14 | −0.007 | −0.007 | +0.000 | −0.007 |
| weight 1.0, hubs 30% | 18/46 | 33 | **−0.029** | −0.020 | +0.004 | **−0.019** |
| weight 1.0, no suppression | 17/46 | 33 | −0.031 | −0.022 | +0.004 | −0.021 |

Against the resolution floor of **0.0122** measured in M2.3a: every gain is inside
it, and the recall and nDCG *losses* at weight 1.0 are outside it. That is the
uncomfortable version of the result — not "no evidence of help" but evidence of
harm, at the only weights where the ranking does anything.

**At 0.5 the ranking is arithmetically inert, and that is not a corpus fact.** RRF's
curve is flat by design: a graph-only candidate at rank 1 contributes 0.5/61 =
0.0082, while a vector-only candidate at rank 30 contributes 1/90 = 0.0111 — and
the vector leg returns fifty of them. Placing a graph-only candidate above vector
rank *r* needs weight > 61/(60+r), so above rank 10 it needs 0.87. Any weight low
enough to be safe is too low to introduce anything, and the safety argument for
0.5 was therefore an argument for a ranking that cannot act. **An introducing
ranking and a rank-flattening fusion are in tension, and this is where that shows
up.** Expansion contributed candidates for 18 of 46 queries and not one of them
reached the top ten at 0.5.

Per query at weight 1.0, 13 of 46 moved — 9 worse, 4 better. The four largest
losses are all recall: a question about two workers and one job (−0.281 nDCG,
−0.429 recall), one about what was worked on around the time of the chunking work
(−0.243, −0.333), one about a long-running job holding its claim (−0.177, −0.250),
and one about cleaning text before splitting it (−0.113, −0.250). In each case the
expansion placed a connected-but-wrong memory above a relevant one.

The largest gain is the query about what the parser registry touches: **+0.064
nDCG, +0.083 recall**, and it is the graph working exactly as designed.
`parsers/pdf.py` entered at rank 5, correctly relevant, found by **neither
retriever** — reached through `ParsedDocument -> PdfReader`, an entity it shares
with the registry. That is an answer no index could have produced. It is also one
query out of forty-six.

The queries themselves are not quoted here, and that is not squeamishness:
`tests/unit/test_golden_hygiene.py` fails the build when a golden query appears
verbatim in a tracked file, because a file holding the query becomes a lexical
match for it and the benchmark starts measuring its own footprint. It caught this
section.

**The confound, stated plainly.** Entity extraction has reached 21 of 162 memories
(13%). Groq's free tier caps `llama-3.3-70b-versatile` at 100,000 tokens per *day*
and this corpus needs roughly 730,000 to extract; `llama-3.1-8b-instant` is bound
at 6,000 tokens per minute, over-extracts at 6.1 mentions per chunk against the
70b's 4.7, and truncated its JSON on 12% of memories; `openai/gpt-oss-20b` spends
its entire completion budget on reasoning tokens and returns no text at all. So the
graph can see a fifth of the corpus, and three of the five queries written for it
saw no expansion because their answers are in the unextracted four fifths. **A
larger graph could change the sign of these numbers. It could not change the
arithmetic above.**

### Neo4j cannot walk from an entity back to itself

Found by a test comparing the Cypher against the in-memory store's breadth-first
walk. The most valuable expansion there is — *another memory mentions the same
thing retrieval just found* — is a path `(seed)-[:MENTIONS]-(m)-[:MENTIONS]-(seed)`,
and Neo4j's relationship-uniqueness rule forbids reusing one relationship inside a
path, so it does not match. The route back only exists through a *different*
memory, which is a longer and weaker connection.

So direct co-mention is matched by its own query rather than by the traversal, and
reported at two hops — the graph distance the path would have had — so that one
scale weights a co-mention against a traversed hop. A `target <> seed` filter that
looked like it was removing a degenerate self-path was in fact removing every
direct co-mention in the corpus.

## Time

M4.0 makes the bitemporal data queryable. `application/temporal.py`, five functions over a
session factory, no ranking change and no UI.

```bash
memoryos timeline --period month           # activity per period, by when things happened
memoryos timeline --period day --source self
memoryos gaps --min-days 14                # silences with activity either side
memoryos as-of 2026-08-10T16:25:00Z        # what the system had ingested by then
```

The same layer has a view in the UI — see [The timeline view](#the-timeline-view).

**Two clocks, and every question here is about which one.** `occurred_at` is when the thing
happened in the world; `ingested_at` is when this system learned about it. `memories_in_range`,
`activity_by_period` and `find_gaps` read the first — it is the clock a person means by "when".
`as_of` reads the second, and only the second — it is the clock a *debugger* means by "when".
`out_of_order` reads both, because the distance between them is itself the signal.

**Why this milestone is a query and not a migration.** M1.1 stored the two columns and recorded
in `occurred_at_source` how each was derived, six milestones before anything read them. Adding
the column today would have been an afternoon. Recovering the values it should have held would
have been impossible: a source moves, a file is rewritten, and the mtime that would have said
last March says today.

### What temporal signal this corpus actually has

Everything below is only as good as this, so it prints above every histogram rather than living
in a doc nobody opens.

```
occurred_at provenance
  filesystem     162  100.0%  2026-08-07 .. 2026-08-10
  unknown          0    0.0%  -

backfill lag over 1d: 88 of 162 memories
  longest 2d 17h  src/memoryos/logging.py
```

| | |
| --- | --- |
| current memories | 162, one source, 160 `code` and 2 `note` |
| `occurred_at_source` | `filesystem` for all 162. No `declared`, no `parsed`, none unknown |
| `occurred_at` span | 2026-08-07 22:28 to 2026-08-10 16:35 — **2 days 18 hours** |
| distinct days | 4, none empty: 14, 50, 33, 65 |
| `ingested_at` span | 2026-08-10 16:15 to 16:35 — **one 20-minute window** |
| backfill lag over 1 day | 88 of 162, longest 2d 17h |

**The honest reading: there is almost no temporal signal here, and what exists is the weakest
kind.** Every date is an mtime, which is `filesystem` provenance for a reason — it records when
a file was last written on this disk, not when the work happened. The corpus is this repository,
written by one person over one week and read once, so the mtimes cluster into the three days
that work happened and the ingestion timestamps into the twenty minutes it took to read them.
Nothing was declared by a source, and no month has a neighbour to be compared against.

The backfill line is the same fact from the other side, and it is a case where the number is
real and the interpretation is not. 88 of 162 memories have `occurred_at` more than a day
behind `ingested_at`, which in a corpus of emails or commits would mean most of it was imported
from somewhere older. Here the longest lag is **2 days 17 hours**, which is the age of the
repository. Nothing was backfilled; the whole corpus was simply read at once, a few days after
it was written. `out_of_order` is measuring correctly and there is nothing here to find — which
is why the threshold is the parameter, and why the *size* of the lag matters more than the count
above it.

So the layer is exercised rather than demonstrated. Every function returns correct answers about
a corpus that has very little to say, and the tests supply the shapes the corpus does not have —
a month boundary, a 39-day silence, a revision ingested after the query time, an undated memory.
That is the right division: the corpus proves the queries run against real data, and the fixtures
prove they are right.

### The timeline has shape, and only at the day grain

```
162 memories by month, 2026-08-07 to 2026-09-01 (1 periods, 1 non-empty)
  2026-08-01     162  ##############################################

162 memories by day, 2026-08-07 to 2026-08-11 (4 periods, 4 non-empty)
  2026-08-07      14  ##########
  2026-08-08      50  ###################################
  2026-08-09      33  #######################
  2026-08-10      65  ##############################################
```

One bar at the default grain — the whole corpus is inside one calendar month, so `--period month`
can only ever draw a single spike. At `--period day` there are four bars and they differ by a
factor of four and a half, which is the shape of somebody's week rather than an artifact. It is
also the finest grain this data supports: below a day the mtimes fragment into fifteen hourly
buckets separated by sleep, which is a chart of when a person was awake.

### Nulls are excluded, never defaulted

An unknown date is not evidence of any date. `occurred_at IS NULL` is in no range and in no
bucket, and the domain already refuses to let it mean anything else — `Memory` raises unless a
null timestamp is paired with `TimeProvenance.UNKNOWN`, and `ck_memories_occurred_at_provenance`
says the same to every other writer.

The tempting implementation coalesces the missing value to `ingested_at` so that every memory
lands somewhere. It would stack every undated memory onto the day the corpus was read and draw
a spike that no event produced — and the spike would be indistinguishable from a real one,
which is what makes it worse than a hole. The undated band is counted in the provenance profile
instead, where a corpus with a large unknown share says so rather than looking smaller.

Ranges are **half-open**, `[start, end)`, for the same reason: closed at both ends puts a memory
timestamped midnight on the 1st into two consecutive months, and a histogram whose bars sum to
more than the corpus is one nobody can reason from.

### `as_of` is reconstructed, not read off `is_current`

The function people skip and later need. Without it "why did this query return that last
Tuesday" has no answer: the corpus has moved on, a ranking is reproducible only against the
inputs it actually had, and a retrieval bug reported against a corpus that no longer exists
cannot be re-run.

`is_current` is a fact about **now**. At a past instant the current version of an item was
whichever version had been ingested by then, and the row wearing the flag today may not have
existed. So `as_of` takes the newest version per `(source_id, external_key)` at or before the
query time — and filters deletion *after* picking the version, because the tombstone updates a
column rather than appending a row, so a memory deleted last week was still known the week
before. Doing both in one pass erases an item retroactively from every past view.

**What it does not reconstruct, stated here rather than discovered later:** the retrieval state.
Chunks are deleted and rewritten in place by re-chunking, embeddings carry only the time they
were last computed, and extraction records a version rather than a history. The *text* the
system held at a past instant is recoverable from this; the ranking it would have produced is
not. A view that quietly claimed the second would be worse than no view at all.

On this corpus the whole ingestion fits in twenty minutes, so `as-of` is a step function with
one step in it — 0 memories before 16:15, 151 at 16:25, 162 after 16:35. That is a real
measurement of a corpus read in one pass.

### Gaps, and the one that is not a gap

A gap is a stretch with activity before it and after it and none during. **This is the
capability the phase exists for.** "When did I abandon this" has no document to retrieve, because
abandonment is never written down — it is the absence of anything after a point, and an absence
is invisible to every retriever here. Vector search finds text that means what the question
means, keyword search finds text that says it, and neither can return a document nobody wrote.
Only aggregation over time can see a hole.

The stretch since the newest memory is deliberately **not** reported. It has activity on one side
only, so nothing distinguishes "abandoned" from "still going, nothing written lately" — and the
open-ended version would fire on every source in every corpus, every time.

At `--min-days 14`, this corpus has none, and could not: it is 2.75 days long. Lowering the
threshold far enough to find anything finds three silences of 10 to 11 hours each, bounded at
22:46→09:31, 20:44→08:13 and 17:52→05:27. **Those are artifacts — they are nights.** Reported
here because the honest version of "gap detection works" on a corpus this shape is that it
correctly finds the only interruptions present, and they are somebody sleeping. The synthetic
39-day gap in the tests is what demonstrates the capability.

### No index was added, and the M1.1 index is why

`EXPLAIN ANALYZE` on the range query, against the real corpus:

```
Seq Scan on memories  (cost=0.00..19.54 rows=82) (actual time=0.004..0.027 rows=83)
  Filter: (is_current AND occurred_at >= ... AND occurred_at < ...)
  Buffers: shared hit=17
```

The planner ignores `ix_memories_occurred_at` and it is right to: the table is **17 pages**.
Forcing the index with `enable_seqscan = off` costs more and reads more — `cost=9.02..27.29`,
18 buffers instead of 17 — because a bitmap scan still has to visit every heap page that holds
a matching row, and at this size that is all of them.

Since "small corpus" is not an argument about the schema, the composite index M4.0 was scoped to
consider was measured at a size where an index certainly is used — 200,000 rows, one month
selected out of thirty:

| indexes available | planner's choice | cost |
| ----------------- | ---------------- | ---- |
| `(occurred_at)` only | bitmap scan on it | 1865.52 |
| `(is_current, occurred_at)` only | bitmap scan on it | 1887.80 |
| both | **`(occurred_at)`** — the composite declined | 1865.52 |

**The composite loses even where indexes win, and it loses on its leading column.** `is_current`
is true for about 95% of rows, so leading with it buys no selectivity and costs a wider key;
the two indexes are the same size and the composite's estimate is 1% worse. Postgres declined
it when offered both. Adding it would have been a migration whose only effect was a second index
to maintain on every write.

The crossover for the *existing* index is somewhere between 169 and 2,000 rows — at 2,000 the
planner already prefers it. So the M1.1 index handles range queries at every size this corpus
might reach, and needs no help. The aggregate behind `activity_by_period` is a sequential scan
at any size, correctly: a histogram over the whole corpus reads every row by definition, and no
index can improve a query that has no rows to skip.

This is the M1.6 lesson applied a second time. An index is not used because it exists, and the
planner is usually right about that.

### The timeline view

M4.1 puts the temporal layer on screen. Three endpoints over `application.temporal` and nothing
else — `/timeline`, `/memories/at`, `/gaps` — and a `timeline` tab beside search.

```bash
make dev                                   # API on :8000, UI on :5173
open http://localhost:5173/timeline?period=day
```

**No chart library.** The dependency would be several hundred kilobytes to draw rectangles whose
heights are a division, and it would bring its own opinions about typography, tooltips and colour
that would then have to be fought. What is needed is one flex row, one percentage per bar and a
`<button>` — which is also how the chart ends up keyboard-navigable for free (arrow keys walk the
periods, Home/End jump to the ends, enter opens one).

Three things the chart says that `memoryos timeline` cannot:

| | |
| --- | --- |
| **stacked by kind** | "Forty memories" and "forty memories that are all code" are different facts, and only one of them is on screen in a terminal. |
| **empty periods hatched** | `find_gaps` exists on the argument that absence is the signal. An absence drawn as absence is indistinguishable from the edge of the data or from a chart that failed to load. |
| **gaps as objects** | In their own lane, positioned by real time rather than snapped to bucket edges — a gap running from the 7th to the 46th does not begin where a bar begins. |

Bars span their period's full share of the axis, so the chart is a real time axis rather than a
row of evenly-spaced columns, and the gap lane below lines up with the hollows it explains. The
fills are washed to 70%: at full strength a wide bar is a field of saturated colour, which is a
dashboard, and this interface spends its one accent on the matched-span highlight.

### Provenance reaches the surface

Every date in the UI now carries how it was derived. `DateStamp` marks anything inferred with a
`~`, anything undated with a `?`, mutes both, and explains itself on hover; **a stated date gets
no mark at all**, because marking every date makes the mark invisible.

```
filesystem~  162 (100.0%)   undated?  0

Every date here is inferred rather than stated. Nothing in this corpus declared when it
happened, so the chart below shows when files were last written to this disk — which is
not the same as when the work happened.
```

That caption is not decoration: it changes what every bar under it means, and it sits above the
chart rather than behind a tooltip for that reason. The tiers are in `web/src/lib/provenance.ts`
— `declared` and `parsed` are claims the source made, `filesystem` and `inferred` are claims we
made about it, and an unrecognised value is treated as the lower tier rather than the higher one.

This is why `occurred_at_source` was added to the search hit as well. A client that received only
the timestamps could not render them differently however much it wanted to.

### Versions on a rail

The memory detail page draws its versions rather than listing them: a hollow tick where the
version occurred, a filled one where it was ingested, and **the rule between them is the lag**
that `out_of_order` measures, made visible per item. Five revisions in an hour and five over a
year are the same five rows in a table and two obviously different pictures here.

What changed is read off the two hashes the corpus already stores. `content_hash` differing with
`normalized_hash` identical — a trailing newline, a line ending, a byte the normalizer discards —
renders as **"bytes only"** rather than as "changed", which is the difference between a version
that changed the corpus and one that only changed the file.

### Two bugs that only running it would find

Both were correct code producing a misleading screen, which is the class of defect a UI milestone
exists to catch.

**The bucket boundaries are UTC and the timestamps were local.** `date_trunc` runs in UTC
deliberately, so the same corpus buckets identically on every machine; `format.timestamp` renders
in the reader's zone deliberately, so a timestamp can be correlated against a log line. Together,
a bar labelled `2026-08-07` listed memories dated `08 Aug 2026, 03:58`, and that reads as an
off-by-one in the bucketing. Views that show bucket boundaries now show their contents in the
zone the boundaries were computed in, and the window is labelled `(utc)`.

**The count labels floated at the plot's ceiling.** Absolutely positioned against the full-height
button rather than against the bar, so every number had to be matched to its bar by eye — which
is exactly the reading error the number was added to remove.

### Evolution, and what a corpus of one clone can show of it

M4.2 reconstructs how one item changed. It is nearly free, and specifically because of
M1.1: a modified file produces a new artifact, a new event and a new memory version, and
the old version keeps its bytes, its normalized text and both hashes. That history has
been in the database, unread, since the first sync.

```bash
memoryos evolution self README.md          # each version, what changed, what it touched
memoryos evolution self README.md --no-summary
```

```
GET /memories/{id}/evolution               # consecutive diffs, cached summaries only
GET /memories/{id}/evolution?from=&to=     # one specific pair
GET /memories/{id}/evolution?summarize=true
```

**How much history this corpus actually has: seven items, two versions each.** 155 of the
162 current memories have exactly one version and no evolution to show. All seven pairs
have different normalized text, so the adoption case — a version whose chunks were moved
across because only the bytes changed — occurs **zero times in the real corpus** and
exists here only in tests. The feature is correct and almost entirely unexercised, which
is what a corpus assembled by cloning a repository once will do.

`difflib`, no dependency, line-level with exact character offsets recovered by summing
line lengths. Two decisions worth naming: `SequenceMatcher` over *characters* on a
54,000-character README finds thousands of one-character runs and produces something
technically correct and unreadable; and `autojunk=False` is not a tuning knob, because
the heuristic discards any element appearing in over 1% of a long sequence and in source
code the commonest lines are blank ones — with it on, a diff of two long files loses its
anchors and reports one enormous replacement.

**The diff is over normalized text, never bytes.** A file rewritten with CRLF is a
genuinely new artifact with a completely different content hash, and it diffs to nothing.
That is the correct answer and it is a live check on M1.4: if normalization ever stops
collapsing line endings, `test_evolution.py` fails on its first test rather than
retrieval degrading three milestones later.

#### Two things this layer refuses to claim

`NormalizeMemory._store` deletes the chunks of every earlier version when it writes the
new ones, because chunks belonging to a version nobody can retrieve stay in the vector
index and keep surfacing stale text. Two consequences follow, and both are reported as
absences rather than papered over:

- A **chunk-count delta** against a superseded version is `n/a`, not a number.
  `after.chunks - 0` would print `+50` for a two-line edit.
- **Affected chunks** are the newer version's only. There is nothing to say about the
  older one's, because they do not exist.

Adoption is still recoverable after the fact: two consecutive versions sharing a
`normalized_hash` are exactly the condition M1.4 acted on, so `evolution` reports "chunks
adopted" rather than showing an empty diff and leaving a reader to wonder whether the
diff failed.

#### Grounding a summary when there is nothing to cite

M2.6's guardrail applies here, and it had to be adapted: a change summary has one piece
of evidence, so there is nothing to number and citation markers would be noise. Three
layers instead.

**The trivial case never reaches the model.** "No substantive change" is decided from the
spans, in code. A model shown an empty diff and asked what changed will answer, because
answering is what it is for, and the answer will be fluent and invented. The prompt asks
for the same string as a second line of defence for near-trivial diffs, not as the first.

**The prompt forbids the rationale.** Rule 2 is "do not explain why the change was made" —
that is the fabrication this milestone is most exposed to, and it is the one no
mechanical check can catch, because "refactored for clarity" names nothing verifiable.

**`check_summary` verifies vocabulary.** Every identifier the summary names must appear
in the diff it was shown. Absent entirely is a fabrication, and it is the exact analogue
of citing passage [7] when six were supplied.

#### What the check caught, and what it did not

The first summary this system generated, on `src/memoryos/config.py`, passed the
vocabulary check and contained a false claim: that two import lines had been *reordered*.
Those lines appear in the diff only as unchanged context. Nothing moved.

Two fixes came out of it. The prompt gained rule 4 — `+` was added, `-` was removed,
everything else is context, never say an unchanged line moved — and the check now splits
the diff into changed lines and context lines and reports terms found **only in context**
separately. Not an error, since "adds a field to `Settings`" is a correct sentence about
a class declared on a context line, but it is precisely where this class of false claim
lives, so it is surfaced rather than passed. After the fix the same pair summarised
cleanly.

The limit is real and worth stating. On the README pair the model wrote that the change
added "details about citations, synthesis, and refusal rates" — and *synthesis* appears
only on a **removed** line. The direction is wrong and the vocabulary check cannot see
it: "synthesis" is a lowercase prose word, not identifier-shaped, and it is genuinely in
the diff. The check is a floor, not a proof.

`summarizer_version` is part of the cache key for exactly this reason, and it caught its
own lesson immediately: rule 4 was added without bumping it, so every read kept serving
the pre-fix text and the fix looked like it had not worked. The same run found that
`--refresh` regenerated the summary, paid for the call, and left the old row in place —
`ON CONFLICT DO NOTHING` was right for the concurrent case and wrong for the explicit
one.

#### Replay classification

`change_summaries` is derived, decided by its foreign keys the same way `entity_merges`
is: both ends point at `memories`, which a replay truncates. It carries a different
discomfort though — nothing in it is anybody's judgement, but every row cost a model call
and a rebuild throws all of them away. It is also the one derived table whose *input*
survives a replay perfectly: the versions come back with identical normalized text, so
the diffs are unchanged and only the descriptions are lost. Keying the cache on the pair
of normalized hashes rather than on memory ids would make these survivable, exactly as
M1.7 proposed for merges. That is a schema change, so it is written down rather than done
quietly.

### Time-aware retrieval

M4.3 lets a query express time, and measures whether that helps.

```bash
memoryos search "what changed in the chunker in August" --explain
MEMOS_TEMPORAL_INTENT_ENABLED=false memoryos search "..."   # the control arm
```

`domain/temporal_intent.py`, rules over a regex, **not a model**. A completion per
search would cost a round trip, make retrieval non-reproducible — the same query
parsing differently on two runs, with nobody able to say why a result moved — and
buy nothing, because the thing being detected is a closed set of English phrases
that fits on one screen.

**The hard part is refusal.** A month name only counts with a temporal preposition
in front of it, because `may` is a modal verb before it is a month, `march` is a
verb, and `august` is an adjective. Which preposition also decides the *bounds*:
`in August` is the month, `since August` is everything from its first day onward,
`before August` is everything up to it, `after August` starts where it ends.
Collapsing those four would look correct on a corpus that fits inside one month
and answer a different question on any corpus that did not.

Three mechanisms, doing three different things:

| intent | example | what happens |
| --- | --- | --- |
| **range** | `in August`, `on 8 August` | hard filter on `occurred_at` — a question about a period is not answered by a document from another one, however relevant |
| **relative** | `recently`, `lately` | recency weight raised **for that query only**; there is no boundary to cut at, and inventing one would drop the answer whenever the guess was wrong |
| **ordering** | `the first version`, `the latest change` | re-sorts the top k by date. The top k, never the candidate pool — sorting candidates before truncating would return the ten oldest memories in the corpus |

Intent `None` takes the M3.5 path untouched, and the test asserts that against the
feature's own off switch rather than a recorded fixture: a fixture proves only that
results match what they matched when it was written.

#### What it measured

Controlled A/B, same code and corpus, parsing off versus on. The committed
`var/baseline-hybrid.json` predates M2.4's reranking, so comparing against it would
mix two changes — the control is `MEMOS_TEMPORAL_INTENT_ENABLED=false` on the same
52-query set.

| metric | off | on | delta | vs the 0.0122 floor |
| --- | --- | --- | --- | --- |
| recall@10 | 0.7726 | 0.7750 | +0.0024 | inside |
| precision@10 | 0.4481 | 0.4500 | +0.0019 | inside |
| MRR | 0.7739 | 0.7904 | **+0.0165** | **outside** |
| nDCG@10 | 0.7268 | 0.7353 | +0.0085 | inside |

**0 of 46 non-temporal queries changed on any metric.** That is the number this
milestone is really about, and it is a pass/fail rather than a delta.

Per temporal query, and they do not move together:

| query | recall | MRR | nDCG |
| --- | --- | --- | --- |
| day range, conjunctive | **+0.125** | **+0.667** | **+0.333** |
| ordering, earliest | +0.000 | **+0.857** | **+0.383** |
| month range | 0 | 0 | 0 |
| relative | 0 | 0 | 0 |
| trap | 0 | 0 | 0 |
| ordering, latest | +0.000 | **−0.667** | **−0.273** |

**The month range moves by exactly zero, structurally.** The whole corpus occurred
between 7 and 10 August 2026, so a filter for August 2026 admits all 162 current
memories. It is a no-op by arithmetic, not by coincidence, and it is the clearest
single statement of what a three-day corpus can show about time.

**The day range is the one filter that acts**, and it acts correctly: the module
implementing the worker carries a 10 August mtime while everything else about the
job queue carries an 8 August one, so a question about the 8th removes a memory
that is topically relevant and from the wrong day. That is what a range filter is
for.

**Earliest ordering worked and latest ordering regressed**, for the same reason.
The oldest job-queue memories *are* the answer to "the first version of the job
queue", so the date sort lifted `0003_jobs.py` and `job_queue.py` from ranks 7 and
8 to 1 and 2. The newest memories in this corpus are `README.md` and `cli.py` — the
two files most likely to be in any top ten and almost never the specific answer —
so latest ordering promoted them over `domain/explanation.py` and halved the MRR.
**The asymmetry is a corpus fact rather than a mechanism fact**, and it is the
argument for the off switch shipping alongside the feature.

**The relative query changed the ranking and improved nothing.** Recency at weight
0.5 reordered the top ten and pulled `README.md` to first — and recall stayed at
0.000 in both arms, because none of the six judged answers was in the candidate
pool to begin with. Signals rank candidates and never introduce them, by design, so
**a weight cannot rescue a query the retrievers failed.** That is the honest reading
of the one result that looked most promising in the prediction.

The global grid still says recency ≈ 0. `tune-weights --grid coarse` over all 52
queries finds a best gain of +0.0013 nDCG, well inside the floor — so M2.3b's
finding survives the addition of six temporal queries, and query-conditional time
is a genuinely different mechanism from a global recency weight rather than a
rebranding of one.

#### The over-eager parse, which was mine

The measurement caught the failure mode it was designed to catch. `time` was in the
list of nouns that make `first`/`latest` temporal, so **"how does the system know a
file changed since last time" parsed as *ordering: latest***, and the date sort
dropped its nDCG from 0.963 to 0.868. "The last time" is idiomatic English for "the
previous occasion" far more often than it is a request for the newest thing, and
the same was true of every loose noun beside it — `thing`, `work`, `state`, `shape`,
`form`. The list is now only nouns denoting a versioned artifact, and the
non-temporal set is untouched.

A query silently reinterpreted as temporal is the most confusing failure available
here, so the interpretation is on `ScoreBreakdown`, on `SearchResult`, in the
search log, and printed above the results:

```
read as temporal: range 2026-08-01..2026-09-01 (from 'in august')  [hard filter applied]
```

It is on `SearchResult` as well as on the breakdown for one specific case: a filter
that excludes the whole corpus returns no hits, so there is no breakdown left to
carry the reason, and "no results" is otherwise indistinguishable from an empty
corpus.

## Phase 4 retrospective

Four milestones: a temporal query layer, a timeline view, evolution and change
detection, and time-aware retrieval.

### Did bitemporal modelling earn the M1.1 decision?

**Yes, and the evidence is that three of these four milestones were query work.**
M1.1 stored `occurred_at` beside `ingested_at` and recorded in `occurred_at_source`
how each was derived, six milestones before anything read them. Adding the column
in M4.0 would have been an afternoon. Recovering the values it should have held
would have been impossible — a source moves, a file is rewritten, and the mtime
that would have said last March says today.

Two specific things would have been unrecoverable rather than merely late:

**`as_of` needs `ingested_at` to be a separate, never-updated column.** "What did
the system know last Tuesday" is answerable only because the ingestion clock was
recorded per version and nothing overwrites it. A single `timestamp` column would
have made past retrieval behaviour unreproducible and therefore undebuggable, and
no amount of later schema work could have reconstructed it.

**`occurred_at_source` is what makes the honest UI possible.** M4.1's whole
contribution is that a filesystem mtime and a date an email declared render
differently. Without the provenance column there is no way to draw that
distinction, and a timeline of mtimes presented as a timeline of events is a chart
that lies confidently. The CHECK constraint pairing a null `occurred_at` with
`TimeProvenance.UNKNOWN` is what stops the null case from being quietly backfilled
by any writer, including psql.

The decision that did **not** pay off is subtler and worth recording: `occurred_at`
was stored but nothing was stored about *how much to trust a range built from it*.
M4.3 discovered that a month filter over this corpus is a no-op, and it discovered
that at query time rather than at ingest time. A column recording the *granularity*
of a date — this mtime is accurate to the second but means "some time that week" —
would have let the range filter widen itself rather than pretending to a precision
the source does not have.

### How much is limited by filesystem mtime being the only date source?

**Almost all of it.** The corpus is 162 memories, 100% `filesystem` provenance,
spanning 2 days 18 hours, ingested in one 20-minute window. Every phase-4 result is
shaped by that:

- **M4.0**: the timeline has one bar at month grain and four at day grain. `find_gaps`
  found three gaps, all of them 10-to-11-hour overnight silences. The capability is
  correct and the corpus has nothing for it to find.
- **M4.1**: the chart's most useful output is the caption above it saying the dates
  are mtimes. Every date on screen carries a `~`.
- **M4.2**: seven items have two versions; 155 have one. The chunk-adoption case
  occurs zero times in the real corpus.
- **M4.3**: one of the two range mechanisms is arithmetically inert, and the
  relative mechanism has no dated answer to promote.

The deeper limitation is not the *range* but the *semantics*. An mtime records when
bytes were last written to this disk. On a fresh clone every file is dated today,
so the entire temporal layer would report one instant — and nothing in the system
would be wrong, because that is genuinely when those bytes arrived. The layer is
measuring the corpus faithfully; the corpus is a poor witness about time.

### What would change with real emails and calendar data?

Four things, in rough order of how much they would change the numbers.

**Provenance would stop being uniform, and the UI would start doing work.** An email
carries a `Date:` header — `declared` provenance, accurate to the second, and a
claim the *sender* made. A calendar event carries a start and an end, which is a
range rather than an instant and would need a column the schema does not have. A
corpus mixing `declared` emails with `filesystem` notes is the first one where
M4.1's `~` marker separates things a reader must not conflate, and where a range
filter has to decide whether a low-confidence date belongs inside the window.

**`find_gaps` would find something.** Abandonment is the capability M4.0 exists for,
and it needs months of activity to detect. A mailbox spanning years has real gaps
with named correspondents on either side — "the last message about this project was
in March, the next one was in November" — which is an answer no retriever can
produce because no document contains it.

**The relative mechanism would have something to promote.** It failed here because
the candidate pool held no dated answer, not because the weight was wrong. Over a
mailbox, "what was I working on recently" has a genuine answer set with real date
separation, and the recency ranking would be reordering candidates that differ by
months rather than by hours.

**`out_of_order` would mean what it says.** Its number here is real and its reading
is not: 88 of 162 memories lag by over a day, and the longest lag is 2d 17h, which
is the age of the repository. Nothing was backfilled. Import a decade of email in an
afternoon and that same query separates the backfilled decade from what arrives
afterwards — which is the difference between a corpus that grew and one that was
assembled, and the thing that distinction is *for* is knowing which timestamps to
trust.

The honest summary of Phase 4: the modelling decision was right, the query layer is
correct, and the corpus cannot exercise it. Those are three separate statements and
the third does not undermine the first two.

## Decisions

M5.0 records what was decided, what else was considered, why, and **what had to be
true**. Four tables plus a review queue, and one rule with an opinion in it.

```bash
memoryos decide --interactive             # the primary path
memoryos decisions list [--status open]
memoryos decisions show <id>
memoryos decisions edit <id> --status settled
memoryos decisions link <id> --evidence self:README.md#8::records
memoryos decisions suggest --limit 10     # propose drafts; commits nothing
memoryos decisions review                 # the queue, with source passages
memoryos decisions accept <id>
memoryos decisions reject <id>
```

### What the corpus actually contained

Phase 5 needs data the first four phases never collected, so the first thing M5.0
did was measure how much of it was already there. Eight decision-shaped queries
through the ordinary hybrid search surfaced 40 distinct memories. A lexical
census over the 142 memories that have chunks found 105 containing a comparative
construction — `rather than`, `instead of`, `trade-off` — and 67 containing one
in the same chunk as a reason. Sampling 28 of those 67 by hand, roughly a quarter
were genuinely decision-shaped and the rest were incidental prose.

**The number that matters is the other one: zero.** Not one memory in the corpus
contains `we chose`, `we decided` or `the decision to`. Fifteen mention an
assumption and none of them means an assumption a decision rested on. No memory
records a confidence, an expected outcome, or a date on which anything was
decided.

So this corpus is dense in *rationale* and empty of *decision records*, and those
are different things. A docstring saying "a table rather than a broker, for two
reasons" names an alternative and gives a reason; it does not say who decided,
when, how sure they were, what they expected, or what they were assuming. Four of
those six fields are what M5.1 and M5.2 read.

**That makes M5.0 a capture problem, not an extraction problem**, and the whole
shape of this milestone follows from it: `decide` is the primary path, `suggest`
is assistive, and every suggestion goes to a queue. The five ADR-shaped sections
of this README — `Hybrid, and why RRF`, `Ranking signals, and why they are off`,
`What it measured, and why the weight ships at zero`, `No index was added, and
the M1.1 index is why`, and the Phase 4 retrospective — are the only passages in
the corpus that come close to a complete record, and even they carry no
confidence.

### A decision has alternatives, or it is a description

`decisions.record` refuses a decision with no rejected option. "We used Postgres"
is a statement in the present tense: there is no counterfactual in it, so no
later outcome can say whether it was right. The rule can only be enforced at
capture — afterwards nothing distinguishes a record whose alternatives were never
written down from one that genuinely had none, and M5.1 would then find that
every decision worked, there having been no other answer to compare against.

The chosen option is written from `chosen` rather than taken from the caller's
list, so the two cannot disagree, and an option whose text equals the choice is
the winner rather than an alternative — which closes the obvious way around the
rule.

### Assumptions are the load-bearing table

An outcome says a decision worked or it did not: one bit, about one decision,
generalising to nothing. An assumption says *why*, and assumptions repeat across
decisions that have nothing else in common. "Deployment will take two days"
failing six times is a pattern with a name and a fix; six unrelated bad projects
is noise with a mood.

`held` and `evaluated_at` are declared now and stay null until M5.2, for the
reason M1.1 declared `occurred_at` six milestones before anything read it. **NULL
means "not yet judged" and is deliberately not `false`** — a system that could
not tell an unevaluated assumption from a broken one would report every new
decision as built on sand.

`decide --interactive` asks for them explicitly, one at a time, and allows
"none". That prompt is doing real work rather than filling a form: asked for
"your assumptions" in one field people write one sentence; asked the same
question five times they produce the third and fourth, which are the ones they
had not noticed they were making.

### `decided_at_source` is M1.1's rule, unchanged

A date somebody typed is `declared`; a date read out of a document is `parsed`; a
date taken from a file's mtime is `filesystem`. Phase 4's weighting applies, and
`decisions list` marks anything that is not `declared` with a `~`, exactly as
M4.1's timeline does. The twelve seeded decisions are all `parsed` — read out of
the milestone each belongs to — because nobody wrote the date down at the time.
Unlike `memories.occurred_at` the column is NOT NULL, so M1.1's null-pairing rule
becomes a prohibition instead: a CHECK forbids `unknown` outright.

### Suggestions are never auto-committed

`decisions suggest` uses the configured `LanguageModel` to propose drafts. It
writes to `decision_suggestions` and never to `decisions`, and that is the whole
safety property.

A model asked to find decisions in explanatory prose will find them, because
prose that explains a choice is shaped exactly like a record of one. What it
cannot find is the half that makes the record worth having — the confidence
somebody held, what they expected, what they were assuming — and asked for those
anyway it produces them, fluently and inventively. That row then becomes a
pattern in M5.3 and a reflection in M5.4, and the resulting claim about how
somebody makes decisions is both plausible and unfalsifiable, because the
evidence for it is a sentence a model wrote.

So the prompt is told to leave `confidence`, `expected_outcome` and `assumptions`
empty unless the passage states them, the review UI prints "not stated" rather
than a blank, and every draft carries the passage it came from. **The queue shows
the passage beside the draft at every width**, because a draft alone always reads
well — accepting has to be a judgement about evidence rather than about
plausibility.

Accepting links the passage as `records`, not `informed`. A design discussion
informed a decision and existed before it; an ADR records it and exists after.
M5.1 needs that ordering, and a pass that marked its own source as an input would
make every extracted decision look as though it had been argued for in advance.

The queue's third button is **edit**, and it is the expected one. It opens the
capture form prefilled from the draft, and submitting there accepts the
suggestion in the same act — accepting first and editing afterwards would leave a
record in the table that nobody stands behind, however briefly. `confidence` and
`expected_outcome` are not carried across even when the model supplied them:
those two are claims about what somebody believed, and starting them from a
model's guess would make the reviewer's job to disagree with a number rather than
to state their own.

**Measured on this corpus.** 25 passages examined, 25 model calls, 8 drafts
queued, 0 unparseable. Of the 8, **4 were worth accepting** — a 50% false-positive
rate among drafts that had already passed the module's own no-alternatives
filter. The four rejected failed in three distinct ways, and all three are worth
naming because they are what a reviewer is actually looking for:

- **A negated choice presented as an alternative.** The shadow-schema foreign-key
  draft offered "Not following derived table references into the shadow schema"
  as the option considered. That is the choice with a `not` in front of it, which
  is a counterfactual nobody weighed.
- **A circular rejection.** An evaluation draft rejected "a guess" because "it is
  not as reliable as an extra search at a wider k" — the reason restates the
  choice.
- **Two passages conflated into one decision.** A README chunk contains both the
  oversized-section rule and the chunk-adoption rule; the draft took the question
  from one and the rejection from the other, producing a record whose reason has
  nothing to do with its question.

And the prediction that held exactly: of 8 drafts, **0 carried a confidence** and
**1 carried an assumption**. The model did not invent them because it was told
not to, and there was nothing in the corpus to find.

### Evidence, and the constraint it collided with

`decision_evidence` links a decision to the memories that informed it, record it,
or contradict it. Its foreign keys into `memories` and `memory_chunks` cascade,
deliberately: a link to a document that no longer exists is a citation to
nothing, and M2.5 spent a milestone making sure a citation always resolves. So
deleting a memory takes its evidence and leaves the decision, which is the right
outcome — a decision is not made false by losing a piece of the evidence for it.

That has two consequences the schema was designed around rather than discovering.

**A full replay truncates `memories`, so `TRUNCATE ... CASCADE` takes this whole
table.** Exactly the trap M1.7 found when the golden set was specified with a
foreign key. The row does not survive; the *link* does. Every row also carries
`(source_name, external_key, chunk_ordinal)`, and `ReplayCorpus._preserve_evidence`
reads the links out before the truncation and re-links them against the rebuilt
corpus afterwards. Measured on this corpus: 30 links preserved, 30 re-linked, 0
dropped, and `decisions list` byte-identical before and after
`replay --from-beginning`.

**And the shadow swap broke.** `decision_evidence` is the first table outside the
derived set to reference something inside it, so `DROP TABLE public.memories` in
`swap_in` now fails on a dependency that has nothing to do with the rebuild.
`DROP TABLE ... CASCADE` would make the error go away by taking the constraints
with it, silently, leaving a live schema that no longer matches the models and an
`alembic check` that fails on the next run. The swap lifts the inbound
constraints off by name and puts them back by definition instead — read off
`Base.metadata` rather than listed by hand, because a hand-kept list goes stale
the first time somebody adds a table and the failure would be a `DROP TABLE`
refusing in the middle of a swap.

### Classification

All five tables are `USER_AUTHORED`, joining `query_judgements`. No amount of
replaying the log produces somebody's account of a choice they made.
`decision_suggestions` is here rather than in the derived set even though a model
wrote its drafts, because the row also carries a person's accept or reject — and,
unlike `entity_merges`, it has no foreign key forcing the classification: its
provenance is a natural key plus id snapshots, so it can be classified by
argument.

### The twelve seeded decisions

`scripts/seed_decisions.py` records twelve real choices from this project's
history — pgvector over FAISS, the Postgres queue over Celery, the bge model swap
and the chunk-offset fix, Groq over Gemini, chunk adoption over per-version
chunking, Neo4j over recursive CTEs, RRF over a weighted sum, ranking signals
shipped at weight zero, graph expansion shipped at weight zero, the golden set's
natural key, and the layering rule left partly unenforced — with the alternatives
that were actually weighed and 35 assumptions between them.

**Their confidences are reconstructions and the script says so.** The corpus
records no confidence for any of them, because nobody wrote one down; the numbers
are what the person who made the call believes they believed. That matters
because it is precisely what M5.2 measures, and a calibration scored against a
number invented afterwards is a calibration of hindsight. Every decision recorded
from here on goes through `decide`, where the number is captured before the
answer is known.

Two of the twelve carry no evidence at all — the Neo4j and graph-expansion ones —
because the corpus is a snapshot taken before Phase 3 and does not contain the
files those decisions are about. That is left as it is rather than papered over
with a nearby file: a decision with no evidence is a real state, the schema
allows it, and the detail view says so.

## Outcomes

M5.1 connects a decision to what happened afterwards. **This is the milestone
where Phase 4 pays for itself**: an outcome is a temporal claim — this occurred
*after* that, close enough to be connected, about the same things — and
`memories_in_range` over `occurred_at`, with nulls excluded rather than
defaulted, is the query that makes it askable at all.

```bash
memoryos outcome <decision-id> --verdict worked --description "..."
memoryos outcomes suggest [--decision ID] [--window-days 90]
memoryos outcomes review                  # gap, window and shared entities stated
memoryos outcomes accept <id>             # writes `inferred`, never `declared`
memoryos outcomes reject <id>
memoryos outcomes rate                    # too_early outside the fraction
memoryos decisions show <id>              # now shows outcomes
```

### `too_early` is a verdict, and it is outside the rate

Most decisions in a young project have no outcome yet. Recording that explicitly
is better than forcing a judgement, and it is a different fact from an absent
outcome: `too_early` means somebody looked and it is too soon to say, an absent
outcome means nobody has looked. The success rate reports three numbers for that
reason — resolved, too early, and never examined — and a corpus with no resolved
outcomes has **no rate at all** rather than a rate of zero, because zero reads
as "everything failed".

### Declared and inferred are different claims

`evidence_kind` is the column M5.3 has to weight by. A declared outcome is
testimony: somebody watched the deployment, read the incident, saw the number
move, and `memoryos outcome` stamps confidence 1.0 because that is what
observing something means — there is deliberately no `--confidence` flag on it.
An inferred outcome is a correlation in time plus a language model's opinion of
it, and a CHECK constraint forbids one from claiming 1.0.

**Accepting a suggestion produces `inferred`, whoever accepts it.** Accepting
means the reading is worth keeping, not that anybody saw it happen, and the CLI
says so on every accept. Without that rule the cheaper kind of evidence — the
kind that scales — would be indistinguishable from the expensive kind, and M5.3
would build its patterns mostly on model output while looking like it was
building them on observation.

### The window, and why it is a guess

There is no correct window. A deployment decision shows its outcome in days; an
architectural one in months. So it is derived per decision from the decision's
own confidence, on the intuition that **a low-confidence decision is one you
expected to learn about sooner**:

```
window_days = 30 + 150 × confidence        # 30 at confidence 0, 180 at 1.0
window_days = 90                           # when no confidence was recorded
```

30 days because below a month a corpus dated by filesystem mtimes cannot
distinguish "after" from "at the same time as"; 180 because past six months
"after" stops implying "because of". A decision with no recorded confidence gets
the flat default rather than a midpoint, because the absence of a number is not
0.5 and pretending otherwise would put an invented confidence into a derived
window.

**It is a stated guess, not a finding, and `outcomes suggest` prints that on
every run.** What would falsify it is M5.2: once enough assumptions have been
evaluated, "low-confidence decisions resolved sooner" becomes a claim the corpus
can answer. `--window-days` overrides it outright, which is the point of writing
a heuristic down rather than burying it.

**And on this corpus it is arithmetically inert.** The whole span is 2 days 18
hours of mtimes, so every window from 30 days to 180 admits exactly the same
memories — the same finding M4.3 made about its month filter, arriving in the
same place for the same reason.

### Four places a candidate is dropped

An outcome suggestion is a worse thing to get wrong than M5.0's. A wrong
decision suggestion proposes a record of a choice nobody made; a wrong outcome
suggestion asserts that one thing *caused* another, and M5.4 would state it as a
fact about how somebody works. Post hoc ergo propter hoc is the oldest error
there is, and a model shown two related-looking documents from one repository
will make it fluently.

So candidates are dropped, never downgraded — a weak candidate among strong ones
teaches a reviewer to skim, which is what the queue exists to prevent:

1. **Strictly after `decided_at`.** M4.0's range is half-open and therefore
   closed at the start, so a memory occurring at the same instant is inside it;
   simultaneous is not afterwards. The database agrees: `gap_days > 0`.
2. **Inside the window**, derived above.
3. **Sharing a resolved entity** with the decision's evidence, followed through
   M3.2's merges so a pre-merge and a post-merge extraction still meet.
4. **Judged by a model allowed to answer `unsure`**, and required to. An unsure
   is a drop, and so is a `yes` below 0.6 — higher than M3.1's 0.5, because a
   wrongly extracted entity is noise in a graph and a wrongly linked outcome is
   a false causal claim.

A decision's own evidence is excluded too. It falls inside the window whenever
its mtime happens to, and admitting it would make every decision look as though
its own reasoning had proved it right.

### `applied` versus `unavailable`, and why the difference is a column

A corpus where nothing has been extracted cannot *fail* the entity test — it
cannot take it. Treating that as "no overlap" returns nothing and looks like a
corpus in which nothing is connected; treating it as overlap silently drops the
constraint and admits every memory in the window. Both are wrong in ways nobody
would notice, so `entity_filter` records which happened on every row and the
queue shows it beside every candidate.

This is not hypothetical. M5.0 ran two full replays, a replay truncates the
entity tables and does not rebuild them, and the corpus spent a whole phase with
**zero** entity mentions — discovered only when M5.1 went looking. `doctor` now
reports it as an advisory, which is what the M5.0 follow-up commit on this branch
added:

```
[note] memories_without_entity_extraction: 162
        current memories no extractor has run over; the graph and any
        entity-scoped query see nothing of them. A full replay truncates the
        entity tables and does not rebuild them — re-run `extract-entities`
        - 0 mention rows in the corpus
```

It happened again during this milestone, which is the point: M5.1's own
verification replay emptied the tables a third time, and this time the check
said so on the next `doctor` run instead of nobody noticing for a phase.

### What this corpus actually says

**Sixteen decisions, twelve with a real outcome and four `too_early`.** The rule
applied while recording them: a decision has an outcome only when its own
`expected_outcome` has been tested. "The code still exists and nothing has
crashed" is not a result.

| verdict | count |
| --- | --- |
| worked | 10 |
| mixed | 2 |
| failed | 0 |
| too_early | 4 |
| no outcome recorded | 0 |

83% of 12 resolved — a number that should be read with its denominator. Zero
failures is not a track record; it is a project a few weeks old whose author
recorded its own decisions, and the four `too_early` rows are the honest part.

**The two `mixed` outcomes are the ones worth reading**, because both are a
decision that achieved exactly what it was for and broke an assumption recorded
beside it:

- **Groq.** The expected outcome held perfectly — provider choice has never
  leaked past `build_language_model`. The assumption recorded at confidence 0.5,
  "the free tier's rate limits are workable for a corpus of this size", did not:
  M3.5 reached 21 of 162 memories against a daily token cap, and M5.1 spent
  fifteen minutes rate-limited to extract twelve.
- **Neo4j.** Postgres-wins-on-disagreement held; nothing writes to the graph but
  the sync. The assumption at 0.45 — "extraction covers enough of the corpus for
  the graph to be dense rather than thin" — broke, and worse than M3.5 knew,
  because the entity tables were then empty for a whole phase.

Both point at the same underlying thing, which is what M5.3 exists to notice and
what this milestone deliberately does not claim to have found.

### What `outcomes suggest` proposed, and why none of it survived review

16 decisions examined, 22 candidates in window, 22 model calls, **4 queued and 0
worth accepting**. A 100% false-positive rate, and the reasons are structural
rather than the model behaving badly — it answered "no" 18 times out of 22,
which is the behaviour the prompt asks for.

Two distinct failure modes, and the second is the one to be afraid of:

- **Same-session sibling files.** Two candidates were `embed.py` and
  `evaluation.py` proposed as outcomes of the shadow-schema decision, **0
  minutes** after it. That decision was accepted from an M5.0 suggestion, so its
  `decided_at` is a file's mtime — and every other file written in the same bulk
  save is seconds later. "Occurred after" here means "was saved later in the
  same working session", which is not a temporal relationship at all.
- **A document that restates the prediction.** Migration `0007_query_judgements`
  was proposed as the outcome of the golden-set natural-key decision, and the
  model's description quoted the decision's own `expected_outcome` back as
  though it were a result. It is the decision *being carried out*, not a report
  of what happened afterwards — M5.0 would have called that `records` evidence.
  This is the dangerous shape, because the words match the expected outcome
  exactly and a reviewer skimming would accept it.

**13 of the 16 decisions had no entity coverage at all**, so their candidates
were found by time alone — which is exactly the state `entity_filter` exists to
make visible, and exactly why every one of those candidates was weak. The three
decisions whose evidence *had* been extracted produced no false positives,
because the overlap test threw the sibling files out before a model ever saw
them. That is one data point and it is the right shape.

The honest conclusion for M5.3: **inferred outcomes contribute nothing on this
corpus.** All twelve real outcomes are `declared`, written by a person checking
what the milestone reports say. A corpus whose entire history is 2 days 18 hours
of filesystem mtimes cannot support the inference — the same limit the Phase 4
retrospective named, reached from a different direction.

### The free tier, hit live

The M5.1 run exhausted Groq's daily token budget mid-milestone — 99,461 of
100,000 on `llama-3.3-70b-versatile` — and the numbers above were produced by
re-running the pass against `llama-3.1-8b-instant`, which has its own budget.
That is the Groq decision's `mixed` outcome happening in real time during the
milestone that records it, and it is why the prompt shows the model 3,000
characters of a candidate rather than 6,000: a measurement nobody can afford to
run is not a measurement.

## Assumptions

M5.2 fills the `held` and `evaluated_at` columns M5.0 declared, and groups the
assumptions that say the same thing across different decisions.

**Assumptions matter more than outcomes, and that is the whole argument for this
milestone.** An outcome tells you a decision worked: one bit, about one decision,
transferable to nothing. "pgvector will be fast enough at my scale" holding or
failing teaches you something you can apply to the next storage decision. "The
pgvector decision worked out" teaches you nothing at all.

```bash
memoryos assumptions review [--decision ID] [--unevaluated]
memoryos assumption <id> --held true|false|partially --note "..."
memoryos assumptions suggest [--decision ID]    # proposes evidence, never a verdict
memoryos assumptions group [--dry-run]
memoryos assumptions candidates                 # pairs the embedder was unsure about
memoryos assumptions accept <id> | reject <id>
memoryos assumptions stats
```

### `held` stops being a boolean

Migration 0018 widens it to `held | failed | partially`, NULL still meaning
nobody has judged it. Forcing a binary produces noise rather than data: "the free
tier's rate limits are workable for a corpus of this size" was true through
months of ordinary use and false the first time a corpus-wide extraction ran, and
recording that as either verdict loses half of what happened.

The column keeps its M5.0 name. `held = 'failed'` reads oddly, and renaming a
column to improve one sentence is how a schema and its documentation drift apart.

`partially` sits **in the denominator of the hold rate and not the numerator**,
which is a judgement rather than an obvious truth — a rate that counted it as a
success would flatter every vague assumption anybody wrote. It is reported on its
own line so the choice stays visible. In the groups view it counts towards
*failure*, and the two rates are deliberately not complements: a belief that half
held is a belief that half broke, and the view whose job is surfacing recurring
trouble should say so.

### The one proposal path with no model in it

`application/assumption_suggest.py` contains no `LanguageModel`, and that absence
is the design rather than an omission. M5.0 asks a model to draft decisions and
M5.1 asks one to judge outcomes, both behind a review queue. Here the retrieval
*is* the proposal: passages that bear on the assumption, with the reason each
surfaced, and nothing that resembles a verdict.

**The system proposes evidence; you judge.** A model asked "did your assumption
hold" produces a fluent guess dressed as an evaluation, and M5.4's reflections
read these values — so a model's opinion here becomes a claim about how a person
thinks, stated as fact and impossible to falsify.

Two filters, and the temporal one is doing real work. A memory that predates the
decision cannot be evidence about whether the belief later held; it is part of
what the belief was formed from, and offering it as a test would be circular.
Measured over six assumptions: 150 passages retrieved, **46 dropped for
predating the decision**. Undated memories go the same way — an unknown date is
not evidence of any date.

### Grouping, and the bar it has to clear

M3.2's machinery over assumption statements, with M3.2's asymmetry intact and a
higher threshold: **0.95 to group automatically, 0.88 to reach the review queue**,
against M3.2's 0.93 and 0.86 for entity names. Assumption statements are full
sentences in one voice about one project, so the whole population sits closer
together than entity names did, and a threshold tuned by intuition would collapse
the corpus into one blob.

A false grouping is worse here than it was for entities. A missed group leaves
two beliefs looking unrelated — visible, and fixed by accepting a pending
candidate. A false group *invents a recurrence*: four members, one hold rate, and
a confident finding about how somebody estimates, assembled from assumptions with
nothing to do with each other. M5.3 reads exactly this table.

When nothing clears the floor the report prints the closest pairs, because "0
groups" does not distinguish a corpus that came close from one nowhere near, and
those call for different next steps.

### What this corpus says

**37 assumptions. 25 evaluated, 12 left alone.**

| | count |
| --- | --- |
| held | 18 |
| failed | 6 |
| partially | 1 |
| unevaluated | 12 |

**Hold rate 72% of 25 evaluated.** The twelve are in neither half, for the reason
`too_early` is outside a success rate: a percentage over whatever happened to get
attention is not a measurement.

The rule applied while evaluating, stated so it can be argued with: **an
assumption is evaluated only when something actually tested it.** "Nothing has
gone wrong" is usually evidence that a belief was never exercised, which is a
different fact from it having held — that is what the twelve are. For a
conjunction, the *load-bearing clause* decides: "`domain/` staying pure is what
protects testability; the rest is tidiness" is judged on the first clause, which
is checkable and checked; "k=60 transfers without tuning, and there is not enough
data to tune it anyway" is judged on the first clause, which nobody tested, so it
stays unevaluated even though the second half is confirmed.

The six failures are worth reading together, because five of them are one story:

- **The free tier's rate limits are workable for a corpus of this size** (0.50) —
  21 of 162 memories at M3.5, and M5.1 exhausted the daily budget outright at
  99,461 of 100,000 mid-milestone.
- **Entity extraction covers enough of the corpus for the graph to be dense
  rather than thin** (0.45) — 13%, then zero, then 7%, then zero.
- **The corpus will contain enough typed relationships for depth-2 traversal to
  reach something a retriever missed** (0.40) — 24 distinct edges.
- **Running a second database is worth it for one query shape** (0.40) — graph
  expansion still ships at weight zero.
- **Someone will re-run this measurement once extraction covers the corpus**
  (0.40) — three milestones later nobody has, and coverage went down.
- **Cosmetic edits are common enough in a real corpus to be worth
  special-casing** (0.60) — M4.2 measured the adoption case occurring zero times.

The single `partially` is the Gemini adapter: "keeping two implementations is
cheap enough that the second one does not rot" (0.60). It has not rotted — it
still typechecks and is still wired in — and nothing exercises it either. When
Groq's daily budget ran out mid-M5.1 the fallback reached for was a *different
Groq model*. Maintained and unproven is neither of the other two verdicts.

### One group, and it is not a pattern

`assumptions group` compared 666 pairs across 37 statements. **Nothing cleared
0.95. One pair cleared the review floor at 0.912** and was accepted by hand:

```
Chunking stays deterministic, so an ordinal identifies the same span after a rebuild.
   from: What do a chunk's char_start and char_end index into?
Chunking stays deterministic, so an ordinal identifies the same span after a replay.
   from: How does a human judgement identify the search result it is about?
```

That is a genuine recurrence — the same belief underwrote two unrelated
decisions — and both members held, giving a group of 2 with a **100% hold rate**.

**It is not a pattern and nothing should treat it as one.** A group of two that
both held says only that determinism held twice. Every other assumption in this
corpus is held exactly once, which is a fact about a project a few weeks old
rather than a failure of the grouper: the five failures listed above *feel* like
one recurring belief about how much of the corpus the language model would reach,
and they are phrased differently enough that 0.88 does not join them and honest
enough that this milestone does not join them by hand. M5.3 gets one group of
two, and that constrains what it can claim.

## Patterns

M5.3 looks for behavioural patterns across decisions. **On this corpus it finds
none, and that is the milestone's result rather than its failure.**

```bash
memoryos patterns discover [--min-support 3]
memoryos patterns list [--kind assumption|timing|choice|outcome]
memoryos patterns show <id>          # statement, evidence, counter-evidence
memoryos patterns dismiss <id> --reason "..."
memoryos patterns calibration        # the table worth reading when nothing emits
```

### The failure mode, stated first

"You consistently underestimate deployment effort" is either a finding backed by
five specific decisions or a horoscope — vague enough to feel true about anyone —
and nothing in the sentence distinguishes the two. A system that produces
confident behavioural claims from thin evidence is worse than one that stays
silent, because it sounds exactly like the product working.

Three gates, each rejecting for a different reason:

1. **Support**, counted in *distinct decisions* rather than rows. Four
   assumptions from two decisions is two observations of one person on two
   occasions; a threshold counting rows would let one decision with five
   assumptions manufacture a pattern about a career. Three is the floor.
2. **Counter-evidence**, found by the same pass that finds the support so it
   cannot be the query somebody forgets. A candidate with more contradicting
   than supporting evidence is **not emitted at all** — not as a weak pattern,
   because a weak pattern still puts the sentence in front of somebody.
3. **Resolution**, for anything arithmetic. A calibration gap is evidence only
   if it exceeds what the sample can distinguish.

Two of those are CHECK constraints rather than conventions in one module:
`support_count > 0` means a pattern that cannot cite is never written, and
`support_count > contradiction_count` means the table itself refuses
confirmation bias.

### The interval is the real gate

M2.3a measured a 0.0122 resolution floor for retrieval and M2.3b refused to ship
a 0.0109 gain because it fell below it. This is the same rule applied to
behaviour, using a Wilson score interval — Wilson rather than the normal
approximation because the normal interval has **zero width at 0 and at 1**, so
fourteen assumptions that all held would report "100%, ± nothing" and every
comparison would find a gap.

**Fourteen out of fourteen gives 78%–100%. A stated 0.85 is inside that.** The
observed gap looks like underconfidence and is not evidence of it: a run of
fourteen cannot distinguish being right 85% of the time from being right always.

Confidence, when a pattern does emit, is derivable rather than assigned:

```
agreement   = supporting / (supporting + contradicting)
sufficiency = min(1, supporting / (2 * min_support))
confidence  = min(0.95, agreement * sufficiency)
```

Three supporting with nothing against scores 0.50, not 1.0 — the least that
counts as anything should not read as certainty. The 0.95 ceiling exists because
a rules-based detector over one person's own records cannot reach certainty
however much agrees.

### What discovery actually produced

**Six candidates, zero emitted.**

```
candidates considered:  6
minimum support:        3 distinct decisions
emitted:                0
below support:          1
within sampling noise:  5

detectors with nothing to propose:
  slow_resolution: no outcome among 12 resolved arrived more than 90 days after its decision
  reversal_rate:   no decision among 16 has been reversed
```

Two of the four detectors produced no candidate at all, and both for reasons
that are facts about the corpus rather than gaps in the code. **Every outcome in
this corpus was recorded on one afternoon**, so the timing detector measures when
somebody sat down to write them rather than when anything resolved. And no
decision has ever been reversed, so the reversal half of the choice detector has
nothing to count.

The one assumption group M5.2 found has two members and both held, so it has
zero *supporting* evidence — a group that held is the opposite of the claim the
detector makes.

### Calibration, which is the honest one

Arithmetic over numbers recorded before the answer was known, needing no
interpretation. Every band, with the interval its sample supports:

| population | band | n | stated | actual | 95% CI | |
| --- | --- | --- | --- | --- | --- | --- |
| assumptions | 0.25–0.50 | 4 | 0.41 | 0% | 0%–49% | within |
| assumptions | 0.50–0.75 | 7 | 0.61 | 57% | 25%–84% | within |
| assumptions | 0.75–1.00 | 14 | 0.85 | 100% | 78%–100% | within |
| decisions | 0.50–0.75 | 2 | 0.55 | 0% | 0%–66% | within |
| decisions | 0.75–1.00 | 6 | 0.89 | 100% | 61%–100% | within |

**Every stated confidence falls inside the interval its own sample supports.**
The shape suggests something — low-confidence beliefs did worse than claimed and
high-confidence ones did better, which would be a person whose uncertainty is
directionally right and poorly scaled — but with four, seven and fourteen
observations that shape is indistinguishable from chance, and saying otherwise
would be this system doing the exact thing it was built not to do.

**And the confound that matters more than any of it.** Calibration is only
meaningful when the confidence was written down before the outcome was known.
Every confidence in this corpus was reconstructed: `scripts/seed_decisions.py`
says so in its own docstring — *"the numbers here are what the person who made
the call believes they believed"*. So the table above is calibration of
hindsight, and it would look exactly like this if it were calibration of
foresight. Nothing in the schema records which it is, and the CLI says so on
every run.

### How many decisions this would need

Not a guess — the interval answers it directly. To distinguish a stated 0.85
from an observed 100%, the lower bound has to clear 0.85:

| n (all holding) | 95% lower bound | 0.85 inside? |
| --- | --- | --- |
| 14 | 0.785 | yes — not a finding |
| 20 | 0.839 | yes — not a finding |
| 25 | 0.867 | **no — a finding** |
| 40 | 0.912 | no |

So **around 25 observations in a single confidence band**, which at roughly two
evaluated assumptions per decision means **50 to 60 decisions with their
assumptions evaluated** — three to four times what this corpus holds. For a
weaker miscalibration the number climbs fast: separating 0.85 from an observed
90% needs several hundred.

The assumption detector is cheaper: three decisions sharing one grouped belief
that mostly broke. M5.2 found one group of two across 16 decisions, so the
binding constraint there is not the threshold but how rarely the same belief is
written down twice.

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
| `POST /decisions`          | Record a decision. Refused if it names no alternative.      |
| `GET /decisions`           | Every decision, with its option/assumption/evidence counts. |
| `GET /decisions/{id}`      | One decision, with everything hanging off it.               |
| `PATCH /decisions/{id}`    | Amend one. Not `confidence`, and not `decided_at`.          |
| `POST /decisions/{id}/evidence` | Link a memory, by its natural key.                     |
| `GET /decisions/suggestions` | The review queue, each draft beside its source passage.   |
| `POST /decisions/suggestions/{id}/accept` | Write a decision from a draft.         |
| `POST /decisions/suggestions/{id}/reject` | Mark a draft as not a decision.        |
| `POST /decisions/{id}/outcomes` | Record an observed outcome. Declared, confidence 1.0. |
| `GET /outcomes/suggestions` | Candidates, with the gap and shared entities stated.     |
| `POST /outcomes/suggestions/{id}/accept` | Write it as `inferred`, never declared.  |
| `POST /outcomes/suggestions/{id}/reject` | Mark a candidate as not an outcome.      |
| `GET /outcomes/rate`       | worked/failed/mixed, with `too_early` outside the rate.     |
| `GET /assumptions`         | Assumptions with decision, outcome, group and evidence.     |
| `GET /assumptions/stats`   | Totals, hold rate, and every group with more than one member. |
| `GET /patterns`            | Patterns with both evidence lists, never one of them.       |
| `GET /patterns/calibration`| Every confidence band with the interval its sample supports.|
| `POST /patterns/{id}/dismiss` | Reject a pattern permanently. A reason is required.      |

There is deliberately no `POST /decisions/suggest` and no `POST /outcomes/suggest`. Running
either extractor costs a model call per candidate, and an endpoint that spends money is one
an over-eager client can spend a daily quota on before anybody notices — the same judgement
`/memories/{id}/evolution` makes about generating summaries on a GET. Both are CLI commands.

`POST /sources/{id}/sync` enqueues and never runs the sync inline. A large directory takes
minutes to walk; doing it in the request would blow the HTTP timeout, and whatever it managed
would have no retry, no progress, and no way to resume.

Liveness stays dependency-free on purpose: a database blip should not cause an orchestrator to
kill otherwise-healthy containers. Readiness is the endpoint that checks dependencies, and
returns `503` when the database is unreachable or `pgvector` is not installed.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (dependency resolution and virtualenv management)
- Docker (for Postgres and Neo4j)

## Quickstart

```bash
make up        # Postgres 17 + pgvector on 5433, Neo4j 5 on 7474 and 7687
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

Integration tests are marked `integration` and need Postgres running. `graph` tests need Neo4j and
skip without it. `slow` tests load the real model and are excluded from the default run.

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
