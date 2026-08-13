"""The real decisions this project made, recorded as decision records.

Not test data. M5.1 through M5.4 read the `decisions` table and nothing else, so
a Phase 5 running over three invented rows would be demonstrating its own
fixtures. These are twelve choices that were actually made while building this
system, with the alternatives that were actually weighed, the reasoning as it was
actually given, and — the part nothing in the corpus contains — the confidence
held at the time and the assumptions the choice rested on.

**Confidence is a reconstruction and is labelled as one.** The corpus records no
confidence for any of these, because nobody wrote one down; the numbers here are
what the person who made the call believes they believed. That is worth stating
because it is exactly the thing M5.2 will measure, and a calibration measured
against a number invented afterwards is a calibration of hindsight. Every future
decision goes through `memoryos decide`, where the number is captured before the
answer is known.

`decided_at` is `parsed` rather than `declared` for the same reason. These dates
come from reading the milestone the decision belongs to out of the README, which
is M1.1's `parsed` exactly: a date recovered from a document, not one somebody
asserted. Phase 4's weighting applies unchanged, and `decisions list` marks them
with a `~`.

Idempotent: a decision whose question is already recorded is skipped, so this can
be re-run after a schema change without doubling the corpus.

    uv run python scripts/seed_decisions.py
"""

import asyncio
from datetime import UTC, datetime

from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.application.decisions import (
    AssumptionInput,
    DecisionDraft,
    EvidenceInput,
    OptionInput,
    UnresolvedEvidence,
    link_evidence,
    record,
)
from memoryos.config import get_settings
from memoryos.container import Container
from memoryos.domain.values import (
    ConfidenceHorizon,
    DecisionStatus,
    EvidenceRelation,
    TimeProvenance,
)

SOURCE = "self"


def when(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, 0, tzinfo=UTC)


def informed(*keys: str) -> tuple[EvidenceInput, ...]:
    """Memories that fed the decision.

    `INFORMED` rather than `RECORDS` where the file is what the decision was
    weighed against — the models file, the port, the measurement. A file that is
    the decision written down gets `records` below, and the two are kept apart
    because M5.1 needs to know which came first.
    """
    return tuple(
        EvidenceInput(
            source_name=SOURCE, external_key=key, relation=EvidenceRelation.INFORMED
        )
        for key in keys
    )


def records(*keys: str) -> tuple[EvidenceInput, ...]:
    return tuple(
        EvidenceInput(
            source_name=SOURCE, external_key=key, relation=EvidenceRelation.RECORDS
        )
        for key in keys
    )


def option(description: str, why: str) -> dict[str, str]:
    return {"description": description, "rejected_because": why}


# --------------------------------------------------------------------------
# The decisions
# --------------------------------------------------------------------------

SEED: list[dict[str, object]] = [
    {
        "decided_at": when(2026, 8, 8),
        "question": "What stores the vectors — Postgres with pgvector, or a dedicated index?",
        "chosen": "Postgres 17 with the pgvector extension",
        "reasoning": (
            "One database means one transaction. A chunk row and its vector are "
            "written together or not at all, and a crash between them is not a "
            "state the system can be in. With a separate index the two are "
            "distinct systems, the window between them is real, and reconciling "
            "them is a permanent background job that exists to repair a problem "
            "the single-store design does not have. The recall cost of HNSW in "
            "Postgres against a specialised index is a tuning knob; the "
            "consistency cost of two stores is architecture."
        ),
        "confidence": 0.85,
        "expected_outcome": (
            "Recall stays above 0.95 at a workable ef_search, and no "
            "reconciliation code is ever written."
        ),
        "options": [
            option(
                "FAISS as a separate index, persisted to disk",
                "No transactional relationship to the rows it indexes; every "
                "crash needs a rebuild or a reconciliation pass, and a stale "
                "index returns confident results for chunks that no longer exist.",
            ),
            option(
                "A managed vector database (Pinecone, Weaviate, Qdrant)",
                "Same two-store consistency problem, plus a network hop and a "
                "vendor in the critical path of a system whose whole point is "
                "that the corpus is private and local.",
            ),
        ],
        "assumptions": [
            ("The corpus stays small enough that HNSW in Postgres is fast enough — "
             "hundreds of thousands of chunks, not hundreds of millions.", 0.8),
            ("pgvector's HNSW recall at a tunable ef_search is close enough to "
             "exact that the difference does not change what a user sees.", 0.7),
            ("Nothing later needs a vector operation pgvector does not have.", 0.75),
        ],
        "evidence": informed(
            "src/memoryos/adapters/db/vector_store.py",
            "src/memoryos/adapters/db/models.py",
        )
        + records("migrations/versions/0005_hnsw_index.py"),
    },
    {
        "decided_at": when(2026, 8, 8),
        "question": "What runs background work — a table in Postgres, or a broker?",
        "chosen": "A durable job queue as a Postgres table, claimed with SKIP LOCKED",
        "reasoning": (
            "Enqueueing a job and writing the data it refers to happen in one "
            "transaction, so there is no window where one committed and the "
            "other did not. With a broker that window is real, and the standard "
            "fix — the transactional outbox — is a jobs table in the database "
            "anyway, so the broker buys a second system to run and the outbox "
            "on top of it. And `SELECT status, count(*) FROM jobs GROUP BY 1` "
            "is the whole monitoring story; every failure's error and traceback "
            "are queryable with SQL."
        ),
        "confidence": 0.9,
        "expected_outcome": (
            "Throughput is never the binding constraint, because embedding is "
            "orders of magnitude slower than the queue's few-thousand-jobs-per-"
            "second ceiling."
        ),
        "options": [
            option(
                "Celery with Redis or RabbitMQ",
                "Cannot enlist in the Postgres transaction, so an enqueue and "
                "the row it refers to can disagree. The fix is an outbox table, "
                "which is this design plus a broker.",
            ),
            option(
                "An in-process asyncio task queue with no durability",
                "A restart loses everything in flight, and the pipeline is long "
                "enough that something is always in flight.",
            ),
        ],
        "assumptions": [
            ("Job throughput stays in the low thousands per second, well under "
             "what a Postgres table can claim.", 0.9),
            ("The `JobQueue` port stays thin enough that swapping the "
             "implementation later is a day, not a milestone.", 0.7),
            ("Postgres is already a hard dependency, so this adds no operational "
             "surface at all.", 0.95),
        ],
        "evidence": informed(
            "src/memoryos/adapters/db/job_queue.py",
            "src/memoryos/application/worker.py",
        )
        + records("migrations/versions/0003_jobs.py"),
    },
    {
        "decided_at": when(2026, 8, 9),
        "question": (
            "Which embedding model, after M1.6.1 found the chunker sized against "
            "the wrong window?"
        ),
        "chosen": "BAAI/bge-small-en-v1.5, with chunk sizes derived from its 512-token window",
        "reasoning": (
            "M1.4 targeted 640 heuristic words against a model that read 256 "
            "WordPieces, so 89% of chunks were silently truncated before "
            "embedding and distinct chunks sharing a prefix embedded "
            "identically. bge-small keeps the 384 dimensions the column was "
            "fixed at in M1.1, so no migration, and doubles the window to 512 "
            "— which is what makes the chunk sizes derivable from the model "
            "rather than chosen. The real fix is not the model, it is that the "
            "composition root now refuses to start when the chunker can emit "
            "more tokens than the model reads."
        ),
        "confidence": 0.8,
        "expected_outcome": (
            "Retrieval quality improves visibly on the assessment queries, and "
            "the truncation class of defect becomes impossible to reintroduce "
            "silently."
        ),
        "options": [
            option(
                "Keep all-MiniLM-L6-v2 and shrink the chunks to its 256-token window",
                "Fixes the truncation and leaves chunks too small to hold a "
                "complete idea, which trades a silent defect for a loud quality "
                "ceiling.",
            ),
            option(
                "A larger model with more dimensions, e.g. bge-base at 768",
                "Changes the fixed-width embedding column, so it is a migration "
                "and a full re-embed of the corpus for a gain nothing had "
                "measured yet.",
            ),
        ],
        "assumptions": [
            ("384 dimensions carry enough signal for a corpus of this size and "
             "kind.", 0.75),
            ("A 512-token window is large enough that structural chunking rarely "
             "has to split a section.", 0.8),
            ("The startup assertion catches the next window mismatch, so this "
             "class of defect does not recur.", 0.9),
            ("Running the model locally on CPU stays fast enough that nothing "
             "needs a GPU or a hosted embedding API.", 0.7),
        ],
        "evidence": informed(
            "src/memoryos/adapters/embedding/sentence_transformers.py",
            "src/memoryos/container.py",
            "tests/unit/test_window_alignment.py",
        ),
    },
    {
        "decided_at": when(2026, 8, 11),
        "question": (
            "What do a chunk's char_start and char_end index into — the stored "
            "chunk text, or the memory's text?"
        ),
        "chosen": (
            "The memory's text, with prefix_chars stored so the relationship to "
            "the chunk's own content is exact"
        ),
        "reasoning": (
            "M1.4 documented the offsets as indexing into the stored text, which "
            "is true only at ordinal 0: the spans tile the document contiguously "
            "while `content` additionally carries the overlap head borrowed from "
            "the previous chunk. 28% of stored chunk text is borrowed lead-in, so "
            "reading the offsets the documented way mis-highlights most chunks — "
            "and plausibly, because the text it points at is real text from the "
            "same document. Recording `prefix_chars` as a column rather than "
            "leaving it to be derived is the actual fix: a derived value every "
            "reader has to rediscover is one most readers get wrong, and the UI "
            "had to measure the corpus to work out what the offsets meant."
        ),
        "confidence": 0.95,
        "expected_outcome": (
            "Citation highlighting is exact for every chunk, and the invariant "
            "`content[prefix_chars:] == memory.content[char_start:char_end]` is "
            "checkable by a command."
        ),
        "options": [
            option(
                "Redefine the offsets to bound the stored text, including the overlap",
                "Would make a citation point at text the document repeats, so "
                "the same span would highlight in two places and neither would "
                "be where the match was.",
            ),
            option(
                "Strip the overlap from the stored text and keep it only for embedding",
                "The overlap exists so a concept spanning a boundary is complete "
                "in one chunk; removing it from what is stored means the text "
                "that was embedded is not the text that can be shown.",
            ),
            option(
                "Derive the prefix length at read time by comparing against the memory",
                "Every reader has to rediscover it, which is how it was got "
                "wrong in the first place, and it costs a document read per "
                "citation.",
            ),
        ],
        "assumptions": [
            ("Chunking stays deterministic, so an ordinal identifies the same "
             "span after a rebuild.", 0.9),
            ("`verify-citations` is run often enough to catch the next drift "
             "between the offsets and the text.", 0.6),
        ],
        "evidence": informed(
            "src/memoryos/adapters/chunking/structural.py",
            "src/memoryos/application/verify_citations.py",
        )
        + records("migrations/versions/0008_chunk_prefix_chars.py"),
    },
    {
        "decided_at": when(2026, 8, 10),
        "question": "Which language model provider answers questions and extracts entities?",
        "chosen": (
            "Groq, behind the LanguageModel port, with Gemini kept as a second "
            "implementation"
        ),
        "reasoning": (
            "The port came first and the provider second, which is what makes "
            "this reversible: `AnswerQuestion`, the grounding checks and the "
            "citation verifier all take a `LanguageModel` and cannot tell which "
            "one they got. Groq is fast enough that a grounded answer feels "
            "interactive and its free tier is usable without a billing "
            "relationship. Keeping Gemini implemented rather than merely "
            "possible is what stops the port from quietly becoming "
            "Groq-shaped."
        ),
        "confidence": 0.6,
        "expected_outcome": (
            "Provider choice never leaks past `build_language_model`, and "
            "switching costs one environment variable."
        ),
        "options": [
            option(
                "Gemini as the default",
                "Slower per call for this workload, and its free tier's shape "
                "was a worse fit for a corpus that needs many small extraction "
                "calls.",
            ),
            option(
                "A locally hosted model via Ollama",
                "No API key and no rate limit, and a quality floor low enough "
                "that grounding failures would be the model rather than the "
                "prompt — which would make every measurement in M2.6 ambiguous.",
            ),
        ],
        "assumptions": [
            ("The free tier's rate limits are workable for a corpus of this "
             "size.", 0.5),
            ("Nothing downstream needs a provider-specific feature — no "
             "structured-output API, no tool use — so the thin port stays "
             "sufficient.", 0.8),
            ("Keeping two implementations is cheap enough that the second one "
             "does not rot.", 0.6),
        ],
        "evidence": informed(
            "src/memoryos/adapters/llm/groq.py",
            "src/memoryos/adapters/llm/gemini.py",
            "src/memoryos/application/ports.py",
        ),
    },
    {
        "decided_at": when(2026, 8, 12),
        "question": "Is the four-layer dependency rule enforced by a tool, or by convention?",
        "chosen": "By convention and review, with `domain/` the only layer held strictly",
        "reasoning": (
            "The rule as written says dependencies point inward only, and "
            "`application/` in fact imports `adapters.db.models` in a dozen "
            "modules — the SQLAlchemy models are the persistence shape the use "
            "cases query through, and hiding them behind a repository per query "
            "would be a large amount of indirection for no behavioural change. "
            "What is actually load-bearing is that `domain/` imports nothing "
            "from the other three, because that is what keeps the invariants "
            "testable without a database. Enforcing the rest with import-linter "
            "would mean either a wall of exemptions or a refactor nobody has a "
            "reason for."
        ),
        "confidence": 0.45,
        "expected_outcome": (
            "The unenforced half stays a documentation problem rather than "
            "becoming a coupling problem; if it does not, import-linter goes in "
            "with the exemptions written down."
        ),
        "options": [
            option(
                "import-linter in CI with contracts for all four layers",
                "Would fail on day one against the existing `adapters.db.models` "
                "imports, so it ships either as a wall of exemptions — which "
                "enforces nothing while looking like it does — or as a refactor "
                "with no behavioural motivation.",
            ),
            option(
                "Route every query through a repository so the rule is true as written",
                "A repository method per query shape, most of them called once. "
                "The indirection would be larger than the coupling it removes.",
            ),
        ],
        "assumptions": [
            ("`domain/` staying pure is the part that actually protects "
             "testability; the rest of the rule is tidiness.", 0.7),
            ("Nobody adds an `adapters/` import to `domain/` without review "
             "catching it.", 0.5),
            ("The README stating the rule more strongly than the code holds it "
             "will not mislead anybody who reads both.", 0.35),
        ],
        "evidence": informed(
            "src/memoryos/application/ports.py",
            "src/memoryos/domain/entities.py",
            "README.md",
        ),
    },
    {
        "decided_at": when(2026, 8, 9),
        "question": (
            "When a file changes cosmetically, are its chunks re-made per version "
            "or adopted by the new one?"
        ),
        "chosen": "Adopted, keyed on normalized_hash",
        "reasoning": (
            "A file saved with CRLF endings is genuinely different bytes, so it "
            "is genuinely a new artifact and a new memory version — the log has "
            "to record that. But its normalized text is identical, so its chunks "
            "and their vectors are still correct, and re-making them would pay "
            "for a full re-chunk and re-embed to produce byte-identical rows. "
            "Two hash levels is what makes that expressible: `content_hash` says "
            "the bytes changed, `normalized_hash` says the meaning did not."
        ),
        "confidence": 0.85,
        "expected_outcome": (
            "A cosmetic edit costs one artifact row and one memory row, and zero "
            "model calls, end to end."
        ),
        "options": [
            option(
                "Chunk every version independently",
                "Simpler and pays a full re-embed for a line-ending change. On a "
                "corpus that is mostly reformatted-in-place files, that is most "
                "of the cost of the pipeline for none of the value.",
            ),
            option(
                "Normalize before hashing, so a cosmetic edit is not a new version",
                "Loses the fact that the bytes changed, which the ingestion log "
                "is supposed to be the record of. A replay could then not "
                "reproduce what was actually observed.",
            ),
        ],
        "assumptions": [
            ("Cosmetic edits are common enough in a real corpus to be worth "
             "special-casing.", 0.6),
            ("Normalization is stable across versions, so the same text produces "
             "the same hash after a library upgrade.", 0.7),
            ("Adoption never moves a chunk onto a version whose text differs, "
             "because the hash is over the whole normalized document.", 0.9),
        ],
        "evidence": informed(
            "src/memoryos/application/normalize.py",
            "src/memoryos/domain/normalization.py",
        ),
    },
    {
        "decided_at": when(2026, 8, 11),
        "question": "What answers variable-depth relationship queries — Neo4j, or recursive CTEs?",
        "chosen": "Neo4j, as a projection Postgres owns",
        "reasoning": (
            "Fixed one- and two-hop relationships are genuinely fine in SQL and "
            "frequently faster; a join table with the right index beats a graph "
            "database at 'which entities does this memory mention'. The break "
            "comes at variable depth: 'what connects this decision to that "
            "person' is two hops or five and which is not known when the query "
            "is written. In SQL that is a recursive CTE whose readability and "
            "cost both degrade with depth; in Cypher it is `[*1..5]`. Making it "
            "a projection rather than a second system of record is what keeps "
            "the cost bounded — on any disagreement Postgres wins and the graph "
            "is rebuilt."
        ),
        "confidence": 0.5,
        "expected_outcome": (
            "Graph-augmented retrieval measurably improves the queries that are "
            "about connection rather than similarity."
        ),
        "options": [
            option(
                "Recursive CTEs over an edges table in Postgres",
                "No second system to run, and a traversal whose cost is an index "
                "lookup per hop rather than a pointer chase. Rejected on "
                "readability and on how badly the plan degrades past three hops "
                "— though on the corpus this actually has, it would have been "
                "enough.",
            ),
            option(
                "An in-memory graph built per query from the Postgres tables",
                "Fine at this corpus size and a rewrite the moment it is not, "
                "with no query language and no persistence for anything that "
                "later wants to store on an edge.",
            ),
        ],
        "assumptions": [
            ("The corpus will contain enough typed relationships for depth-2 "
             "traversal to reach something a retriever missed.", 0.4),
            ("Entity extraction covers enough of the corpus for the graph to be "
             "dense rather than thin.", 0.45),
            ("Running a second database is worth it for one query shape.", 0.4),
            ("Keeping the graph strictly a projection stops it becoming a second "
             "source of truth nobody can reconcile.", 0.85),
        ],
        "evidence": informed(
            "src/memoryos/adapters/graph/neo4j_store.py",
            "src/memoryos/application/graph_projection.py",
            "src/memoryos/application/graph_expand.py",
        ),
    },
    {
        "decided_at": when(2026, 8, 9),
        "question": "How are two retrievers combined into one ranking?",
        "chosen": "Reciprocal rank fusion at k=60, discarding both retrievers' scores",
        "reasoning": (
            "A weighted sum of cosine and ts_rank_cd does not work and no amount "
            "of tuning fixes it: the two numbers are not on comparable scales. "
            "Cosine from this model occupies a narrow band where almost all the "
            "range carries no signal, and ts_rank_cd is unbounded and depends on "
            "term frequencies across the whole corpus, so the same document "
            "scores differently after ingesting unrelated files. Every "
            "normalisation that would make them comparable encodes an assumption "
            "about the score distribution that stops holding as the corpus "
            "grows, and it fails silently. RRF keeps only the ordering, which is "
            "the part both retrievers mean the same thing by."
        ),
        "confidence": 0.9,
        "expected_outcome": (
            "Hybrid beats both halves on the golden set, and the fused score "
            "stays stable as the corpus grows."
        ),
        "options": [
            option(
                "A weighted sum of the normalised scores",
                "The two scales are not comparable and any normalisation "
                "encodes an assumption about the distribution that decays "
                "silently as the corpus grows.",
            ),
            option(
                "Cross-encoder reranking alone, with no fusion",
                "A reranker cannot recover what retrieval missed — a document "
                "outside the shortlist cannot be ranked into it, however "
                "relevant.",
            ),
        ],
        "assumptions": [
            ("Rank is the thing both retrievers agree on, and their scores carry "
             "nothing extra worth keeping.", 0.85),
            ("k=60 from the original paper transfers to this corpus without "
             "tuning — and there is not enough golden data to tune it against "
             "anyway.", 0.6),
            ("The two retrievers fail in different directions often enough for "
             "agreement to be informative.", 0.8),
        ],
        "evidence": informed(
            "src/memoryos/domain/fusion.py",
            "src/memoryos/application/search.py",
            "tests/unit/test_fusion.py",
        ),
    },
    {
        "decided_at": when(2026, 8, 10),
        "question": "Do recency and importance ship as ranking signals?",
        "chosen": "Shipped, wired into fusion, and defaulted to weight zero",
        "reasoning": (
            "The grid search said no. Recency monotonically lowers nDCG at every "
            "importance level tried — 0.735 at weight 0 down to 0.707 at 0.60 — "
            "because on a repository of explanatory prose, when a file was last "
            "edited says almost nothing about whether it answers a question "
            "about the design. The best importance weight gains 0.0109, which is "
            "below the 0.0122 resolution floor M2.3a measured, so it is not "
            "evidence. Shipping the mechanism at zero rather than deleting it "
            "keeps the measurement reproducible on a corpus where the answer "
            "might differ."
        ),
        "confidence": 0.75,
        "expected_outcome": (
            "The weights stay at zero on this corpus, and a corpus with real "
            "date spread moves them off it."
        ),
        "options": [
            option(
                "Ship recency on by default at a small weight",
                "Lowers nDCG monotonically on this corpus. It trades recall for "
                "MRR — at weight 0.6, MRR 0.762 to 0.839 while recall falls "
                "0.852 to 0.779 — which is the wrong trade before citations and "
                "synthesis.",
            ),
            option(
                "Delete the signals and the tuning harness",
                "Throws away the measurement along with the result, and the "
                "result is corpus-specific: the next corpus is a mailbox, where "
                "recency is not noise.",
            ),
        ],
        "assumptions": [
            ("The 41-query golden set is large enough for the 0.0122 resolution "
             "floor to be a real bound rather than an artefact.", 0.65),
            ("A gain below the floor is genuinely not evidence, so shipping it "
             "would be shipping noise.", 0.9),
            ("A corpus with real date spread would move these weights, so the "
             "code is worth keeping.", 0.7),
        ],
        "evidence": informed(
            "src/memoryos/domain/signals.py",
            "src/memoryos/application/tuning.py",
        ),
    },
    {
        "decided_at": when(2026, 8, 9),
        "question": "How does a human judgement identify the search result it is about?",
        "chosen": (
            "By the natural key (source_name, external_key, chunk_ordinal), with no "
            "foreign key to memories"
        ),
        "reasoning": (
            "The milestone specified a foreign key to `memories`. Measured on "
            "this database, `TRUNCATE memory_chunks, memories, jobs CASCADE` "
            "empties any table referencing memories, and a plain DELETE takes it "
            "too via ON DELETE CASCADE — so every routine replay would have "
            "destroyed the golden set, which is precisely what the "
            "USER_AUTHORED classification exists to prevent. Ids are minted per "
            "write and a rebuild legitimately changes them; names do not move. "
            "The ids survive as snapshots of what the system pointed at when "
            "somebody disagreed with it."
        ),
        "confidence": 0.95,
        "expected_outcome": (
            "The golden set survives any number of rebuilds and re-resolves to "
            "current ids on export."
        ),
        "options": [
            option(
                "A foreign key to memories.id, as the milestone specified",
                "Measured: every full replay deletes the golden set, either by "
                "cascade or by blocking the replay outright.",
            ),
            option(
                "Store the memory id with no foreign key and re-resolve on read",
                "Half the fix. The pointer is stale after a rebuild and nothing "
                "in the row says what it used to mean, so an export cannot "
                "recover the item.",
            ),
        ],
        "assumptions": [
            ("External keys are stable enough that a moved file is rare and a "
             "renamed one is worth losing.", 0.6),
            ("Chunking stays deterministic, so an ordinal identifies the same "
             "span after a replay.", 0.85),
            ("Judgements are worth more than the referential integrity a "
             "foreign key would give.", 0.9),
        ],
        "evidence": informed("src/memoryos/application/judgements.py")
        + records("tests/integration/test_judgements.py"),
    },
    {
        "decided_at": when(2026, 8, 12),
        "question": "Does graph expansion ship on, after M3.5 measured it?",
        "chosen": "Shipped at weight zero, with the mechanism and the measurement both kept",
        "reasoning": (
            "It did not pay for itself and the arithmetic says why. RRF is "
            "conservative by design — at weight 0.5 a graph-only candidate "
            "contributes 0.5/61 against a retriever's 0.0164, so a memory "
            "neither retriever found has to be ranked first by the graph to "
            "reach the middle of the fused list. That is the correct default and "
            "it also bounds how much good expansion can do. The confound is "
            "stated rather than argued around: extraction had reached 21 of 162 "
            "memories, so the graph could see a fifth of the corpus."
        ),
        "confidence": 0.7,
        "expected_outcome": (
            "A larger graph changes the sign of the per-query numbers and does "
            "not change the arithmetic about RRF's conservatism."
        ),
        "options": [
            option(
                "Ship expansion on at a moderate weight",
                "No corpus-wide gain above the resolution floor, and the failure "
                "mode is dragging a relevant result out of the top ten on "
                "queries the mechanism was not designed for.",
            ),
            option(
                "Remove graph expansion entirely",
                "The measurement is confounded by 13% extraction coverage, so "
                "removing it would be concluding from an experiment that has not "
                "run.",
            ),
        ],
        "assumptions": [
            ("The 13% extraction coverage, not the mechanism, is what made the "
             "measurement flat.", 0.65),
            ("RRF's conservatism is the right default even when it bounds the "
             "upside.", 0.85),
            ("Someone will re-run this measurement once extraction covers the "
             "corpus.", 0.4),
        ],
        "evidence": informed(
            "src/memoryos/application/graph_expand.py",
            "predictions.md",
        ),
    },
]


async def main() -> None:
    settings = get_settings()
    container = Container.build(settings)
    written = 0
    skipped = 0
    unlinked = 0
    try:
        sessions = container.database.session_factory
        async with sessions() as session:
            existing = {
                row[0]
                for row in await session.execute(select(models.Decision.question))
            }

        for entry in SEED:
            question = str(entry["question"])
            if question in existing:
                skipped += 1
                continue

            options = entry["options"]
            assumptions = entry["assumptions"]
            assert isinstance(options, list)
            assert isinstance(assumptions, list)
            evidence = entry["evidence"]
            assert isinstance(evidence, tuple)

            draft = DecisionDraft(
                question=question,
                chosen=str(entry["chosen"]),
                reasoning=str(entry["reasoning"]),
                confidence=float(entry["confidence"]),  # type: ignore[arg-type]
                expected_outcome=str(entry["expected_outcome"]),
                options=tuple(
                    OptionInput(
                        description=item["description"],
                        rejected_because=item["rejected_because"],
                    )
                    for item in options
                ),
                assumptions=tuple(
                    AssumptionInput(statement=statement, confidence=confidence)
                    for statement, confidence in assumptions
                ),
            )

            decided_at = entry["decided_at"]
            assert isinstance(decided_at, datetime)
            decision_id = await record(
                sessions,
                draft,
                decided_at=decided_at,
                # Read out of the milestone this decision belongs to, which is
                # M1.1's `parsed` exactly. Not `declared`: nobody wrote the date
                # down at the time.
                decided_at_source=TimeProvenance.PARSED,
                status=DecisionStatus.OPEN,
                # Declared rather than left to be derived. `parsed` already
                # implies it — see `domain/patterns.classify_confidence` — but
                # this script is the one caller that *knows*, because its own
                # docstring says the numbers are what the person believes they
                # believed. Saying so here means a future change to the
                # derivation cannot quietly promote these twelve rows into a
                # calibration population.
                confidence_horizon=ConfidenceHorizon.HINDSIGHT,
            )
            written += 1

            # Linked one at a time rather than inside `record`, so a file that
            # has left the corpus costs its own link instead of the whole
            # decision.
            for link in evidence:
                try:
                    await link_evidence(sessions, decision_id, link)
                except UnresolvedEvidence as exc:
                    unlinked += 1
                    print(f"  unlinked: {exc}")
    finally:
        await container.dispose()

    print(f"recorded {written}, skipped {skipped} already present, {unlinked} links unresolved")


if __name__ == "__main__":
    asyncio.run(main())
