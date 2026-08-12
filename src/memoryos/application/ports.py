"""Ports: what the application layer needs from the outside world.

Protocols only. No implementations, no imports from `memoryos.adapters`. The
dependency arrow points inward: adapters implement these, never the reverse.

`async_sessionmaker` is the one infrastructure type that appears here, because
every use case in this layer already takes one as its unit-of-work factory. It is
a type, not an adapter, and pretending otherwise would mean inventing a wrapper
whose only job is to not be named SQLAlchemy.
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.domain.entities import IngestionEvent, Memory, RawArtifact, Source
from memoryos.domain.jobs import Job, JobSpec
from memoryos.domain.values import (
    ContentHash,
    EdgeType,
    EntityType,
    GraphLabel,
    MemoryKind,
    Predicate,
    SourceKind,
    TimeProvenance,
)


class SourceRepository(Protocol):
    async def get(self, source_id: UUID) -> Source | None: ...

    async def get_by_name(self, kind: SourceKind, name: str) -> Source | None: ...

    async def add(self, source: Source) -> None: ...

    async def update_cursor(self, source_id: UUID, cursor: dict[str, Any]) -> None: ...


class ArtifactRepository(Protocol):
    async def exists(self, content_hash: ContentHash) -> bool: ...

    async def add(self, artifact: RawArtifact) -> None: ...


class EventLog(Protocol):
    """The append-only ingestion log.

    There is no `update` and no `delete`, and their absence is the point: the
    log is what `memories` and `memory_chunks` are replayed from, so a rewritten
    event would silently change history that has already been projected.
    """

    async def append(self, event: IngestionEvent) -> IngestionEvent:
        """Append, and return the event as the log recorded it.

        The returned copy carries `seq` and `recorded_at`, which the database
        assigns and the caller cannot know beforehand. That matters more than it
        looks: `recorded_at` is the timestamp every value derived from this event
        must be stamped with, so that replaying the log reproduces them exactly.
        A caller that used its own clock instead would write a projection the log
        cannot regenerate.
        """
        ...

    async def replay(self, after_seq: int = 0, limit: int = 1000) -> Sequence[IngestionEvent]: ...


class MemoryRepository(Protocol):
    async def get_current(self, source_id: UUID, external_key: str) -> Memory | None: ...

    async def add_version(self, memory: Memory) -> None: ...

    async def tombstone(self, memory_id: UUID, at: datetime) -> None:
        """Mark a memory deleted as of `at`.

        The timestamp is passed in rather than taken from the clock, and that is
        a replay requirement: `deleted_at` is a derived value, so it has to be a
        function of the event that caused it — its `recorded_at` — or a rebuild
        produces a corpus that differs from the original in a column nobody
        would think to look at.
        """
        ...


class JobQueue(Protocol):
    """A durable work queue.

    Every mutating method takes `worker_id` and returns `bool`. That is
    fencing: each statement carries `AND locked_by = :worker_id`, so a worker
    whose lease expired while it was busy cannot write terminal state over a
    job that another worker has since claimed. `False` means "you were fenced
    out" — the caller no longer owns this job and must not touch it.
    """

    async def enqueue(self, spec: JobSpec) -> UUID | None:
        """Returns None if a job with the same (job_type, dedupe_key)
        is already pending or running."""

    async def claim(self, worker_id: str, lease: timedelta) -> Job | None: ...

    async def heartbeat(self, job_id: UUID, worker_id: str, lease: timedelta) -> bool:
        """False means the lease was lost — another worker may now own this job."""

    async def complete(self, job_id: UUID, worker_id: str) -> bool: ...

    async def reschedule(
        self, job_id: UUID, worker_id: str, run_after: datetime, error: str, tb: str
    ) -> bool: ...

    async def dead_letter(self, job_id: UUID, worker_id: str, error: str, tb: str) -> bool: ...

    async def reclaim_expired(self, limit: int = 100) -> int: ...


class BlobStore(Protocol):
    """Where artifact bytes live.

    M1.1 stored only hashes, on the grounds that identity is a function of
    content. The bytes still have to go somewhere, and this is the seam that
    lets them go to a local directory now and to object storage later without
    a use case noticing.
    """

    async def put(self, content_hash: ContentHash, data: bytes) -> None:
        """Idempotent. Writing the same hash twice is a no-op."""

    async def put_stream(self, chunks: AsyncIterator[bytes]) -> tuple[ContentHash, int]:
        """Stream bytes to storage, computing the hash in transit.

        Returns the content hash and byte size.

        The hash is not known until the last byte has passed through, so the
        destination path is not known either — which is why this writes to a
        temp file and moves it into place once the name exists. One pass over
        the source instead of one to hash and another to store.
        """
        ...

    async def get(self, content_hash: ContentHash) -> bytes: ...

    async def exists(self, content_hash: ContentHash) -> bool: ...

    async def delete(self, content_hash: ContentHash) -> None: ...


@dataclass(frozen=True, slots=True)
class ObservedItem:
    """One thing a connector found, described without reading it."""

    # Relative to the source root, POSIX-normalised. Never absolute: an
    # absolute path stops being a stable identity the moment the folder moves.
    external_key: str
    content_hash: ContentHash
    byte_size: int
    media_type: str | None
    occurred_at: datetime | None
    occurred_at_source: TimeProvenance
    # Lazy on purpose. Most files on most syncs are unchanged, and reading
    # every byte of every file to discover that would defeat the point of
    # having a change filter at all.
    read_bytes: Callable[[], Awaitable[bytes]]
    # Whatever the connector wants remembered about this item so that the next
    # sync can cheaply decide whether to look at it again — `(mtime_ns, size)`
    # for the filesystem. The sync use case stores it verbatim in the source
    # cursor and never interprets it; only the connector that wrote it knows
    # what it means.
    fingerprint: list[Any] | None = None


class Connector(Protocol):
    """Walks a source and reports what it finds."""

    kind: SourceKind

    def observe(self, source: Source, *, full: bool) -> AsyncIterator[ObservedItem]:
        """Yield items, streaming.

        An async iterator rather than a list, because sources get large.
        Materialising every item before processing any of them works on a
        fixture directory and fails on the first real corpus; streaming keeps
        memory flat whether there are fifty files or five hundred thousand.
        """
        ...


@dataclass(frozen=True, slots=True)
class StructureMarker:
    """A place in the text where the author signalled a boundary."""

    # "heading" | "code_block" | "definition"
    kind: str
    # Heading depth, or 0 where depth is meaningless.
    level: int
    char_offset: int
    label: str | None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """One format's content, expressed the way every format expresses it.

    Everything downstream is format-blind. That is what stops the pipeline
    growing a branch per file type, and it is why the chunker can be written
    once rather than once per parser.
    """

    text: str
    title: str | None
    metadata: dict[str, Any]
    # What the parser already knew about where topics change. Without it the
    # chunker is guessing at boundaries somebody had already marked.
    structure: list[StructureMarker]
    kind: MemoryKind = MemoryKind.OTHER


class Parser(Protocol):
    def can_parse(self, media_type: str | None, external_key: str) -> bool: ...

    def parse(
        self, data: bytes, *, media_type: str | None, external_key: str
    ) -> ParsedDocument: ...


@dataclass(frozen=True, slots=True)
class TextChunk:
    """A retrievable span, with offsets into the document it came from.

    The invariant, exactly:

        text[prefix_chars:] == document[char_start:char_end]

    `char_start`/`char_end` name the chunk's *own* span, and those spans tile
    the document contiguously. `text` is longer than that span for every chunk
    but the first, because the chunker prepends an overlap prefix borrowed from
    its predecessor so a concept crossing a boundary appears whole somewhere.
    `prefix_chars` is how long that borrowed head is.

    M1.4 documented the offsets as indexing exactly into `text`, which is true
    only at ordinal 0. Anything computing a span from that reading highlights
    text the chunk does not claim — and plausibly, since the text it highlights
    is real text from the same document.
    """

    ordinal: int
    text: str
    char_start: int
    char_end: int
    token_count: int
    # Length of the borrowed overlap head. 0 at ordinal 0, which has no
    # predecessor to borrow from.
    prefix_chars: int = 0
    # What the chunker knew about where this came from. A definition split
    # across several chunks records its name here, so a citation can still say
    # which function the span belongs to.
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker(Protocol):
    @property
    def max_tokens(self) -> int:
        """The hard ceiling, in model tokens.

        A hard invariant rather than a preference: exceeding it means text is
        silently discarded before embedding, which is data loss that nothing
        reports.
        """
        ...

    @property
    def version(self) -> str:
        """Identifies the algorithm *and its parameters*.

        Encoding the parameters is what turns "improve the chunker" into a
        query — select the memories whose chunks carry the old version and
        re-chunk only those — instead of a full corpus rebuild. It also means
        that six months from now, a stamp on a chunk that retrieved badly says
        exactly what produced it.
        """
        ...

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]: ...


class TokenCounter(Protocol):
    """How a model counts, and where it stops reading.

    Deliberately narrower than `Embedder`. The chunker needs to size text the
    way the model will read it; it has no business producing vectors, and a
    port that let it would be an invitation. It also means the chunker can be
    tested against a counter without loading a model.
    """

    @property
    def max_sequence_tokens(self) -> int:
        """Tokens the model actually reads. Text beyond this is discarded."""
        ...

    def count_tokens(self, text: str) -> int:
        """Count using the model's own tokenizer, not an approximation.

        M1.4 counted words and punctuation, which is within ~25% on prose and
        out by 2.7x on code, where identifiers split into several WordPieces.
        Sizing chunks against the wrong unit is what let text past the window.
        """
        ...


class Embedder(TokenCounter, Protocol):
    """Turns text into vectors.

    The clearest case in Phase 1 for a port. Phase 2's evaluation harness will
    find a model that does better on this corpus, and that swap has to be a new
    adapter rather than a refactor of everything that touches embeddings.
    """

    @property
    def model_id(self) -> str:
        """Stable identifier including version.

        e.g. 'sentence-transformers/all-MiniLM-L6-v2@1'. It is part of the
        cache key, so it has to change whenever the vectors would.
        """
        ...

    @property
    def dimension(self) -> int: ...

    @property
    def normalizes(self) -> bool:
        """True if output vectors are unit length."""
        ...

    def embed_passage(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text that is being stored and searched over.

        Synchronous on purpose. This is CPU-bound matrix multiplication; an
        async signature would invite someone to await it on the event loop,
        where it would stall every other coroutine in the process — including
        health checks. Callers push it onto a thread.

        There is deliberately no role-free `embed`. Several retrieval models are
        asymmetric, and one that is used symmetrically does not fail — it just
        retrieves worse than it should, which is the same shape of silent defect
        as M1.6.1's window misalignment. Making the role a required part of the
        call means nobody can embed without answering the question.
        """
        ...

    def embed_query(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed text that is being searched *with*.

        Defaults to `embed_passage`, which is correct for a symmetric model.
        An asymmetric one overrides this; see the query prefix in the
        sentence-transformers adapter.
        """
        return self.embed_passage(texts)


@dataclass(frozen=True, slots=True)
class CacheEntry:
    cache_key: str
    model_id: str
    dimension: int
    embedding: list[float]


class EmbeddingCache(Protocol):
    """Text-to-vector memoisation, keyed by (model, text).

    Its own table rather than a column on `memory_chunks`, because identical
    text in different memories should be embedded once. M1.4's smoke test found
    five identical empty files in this repository alone; a real corpus repeats
    far more than that.
    """

    async def get_many(self, keys: Sequence[str]) -> dict[str, list[float]]: ...

    async def put_many(self, entries: Sequence[CacheEntry]) -> None: ...


@dataclass(frozen=True, slots=True)
class SearchFilters:
    source_ids: Sequence[UUID] | None = None
    kinds: Sequence[MemoryKind] | None = None
    occurred_after: datetime | None = None
    occurred_before: datetime | None = None
    # Off by default. A tombstoned memory's chunks resurfacing in results is
    # the exact failure the ON DELETE CASCADE and the deletion guardrail exist
    # to prevent.
    include_deleted: bool = False


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """Where a chunk's score came from. The explainability guardrail.

    Not optional, and not a debugging convenience. Once two retrievers are fused
    into one number, that number is the only thing a reader sees, and it is
    unfalsifiable without this: a result ranked fourth cannot be explained, a
    regression cannot be attributed to a half, and M2.5's citations have nothing
    to show beyond "the system says so". The cost is one dataclass; the
    alternative is a ranking nobody can argue with.

    `fused` is whatever the chunk was actually ranked by — the RRF score under
    hybrid, and the retriever's own score under a single-retriever mode, so the
    field never lies about what produced the order. The per-retriever scores are
    kept raw and uncomparable, because normalising them is precisely what RRF
    exists to avoid.

    A null rank means that retriever did not return this chunk at all, which is
    different from returning it last.
    """

    fused: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    # The signal rankings. They rank the *candidates the retrievers found* and
    # never introduce a chunk of their own — see `application/search.py` — so a
    # null here means the chunk was not a candidate at all, not that it has no
    # date.
    recency_rank: int | None = None
    importance_rank: int | None = None
    recency_score: float | None = None
    importance_score: float | None = None
    # The cross-encoder's verdict, and where it placed this chunk. Null when
    # reranking was off or the chunk fell outside the shortlist — which is the
    # distinction that matters, because a chunk the reranker never saw is not
    # the same as one it scored badly.
    rerank_score: float | None = None
    rerank_rank: int | None = None
    # M3.5's graph expansion. `graph_rank` is where the expansion placed this
    # chunk; `graph_path` is the entity route that reached it, seed entity first.
    #
    # The path is not a debugging aid. Graph expansion is the one ranking that
    # *introduces* candidates rather than reordering them, so it is the one whose
    # contribution a reader is least able to reconstruct — a result that shares no
    # word with the query is inexplicable without the route, and an unexplained
    # promotion is exactly what M2.5 built this dataclass to prevent. A chunk with
    # a `graph_rank` and no `graph_path` would be the same failure wearing a
    # number.
    graph_rank: int | None = None
    graph_score: float | None = None
    graph_path: tuple[str, ...] | None = None
    # M4.3's temporal intent. A property of the *query* rather than of the chunk,
    # and carried here anyway, because this dataclass is the only thing that
    # travels from the ranker to the screen — and a query silently reinterpreted
    # as temporal is the most confusing failure this system can produce. The
    # results change, the reason is invisible, and the user's own words are the
    # only thing that can explain it.
    #
    # `temporal_filter_applied` is separate from `temporal_intent` because they
    # answer different questions. "Recently" is detected and changes a weight;
    # "in August" is detected and *removes rows from consideration*. A reader who
    # cannot see which one happened cannot tell a reordering from a truncation.
    temporal_intent: str | None = None
    temporal_filter_applied: bool = False

    def as_dict(self) -> dict[str, float | int | str | bool | None]:
        return {
            "fused": self.fused,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
            "recency_rank": self.recency_rank,
            "importance_rank": self.importance_rank,
            "recency_score": self.recency_score,
            "importance_score": self.importance_score,
            "rerank_score": self.rerank_score,
            "rerank_rank": self.rerank_rank,
            "graph_rank": self.graph_rank,
            "graph_score": self.graph_score,
            # Rendered as the route a reader would follow, because that is what it
            # is: `queue -> SKIP LOCKED -> worker`.
            "graph_path": None if self.graph_path is None else " -> ".join(self.graph_path),
            "temporal_intent": self.temporal_intent,
            "temporal_filter_applied": self.temporal_filter_applied,
        }


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: UUID
    memory_id: UUID
    ordinal: int
    text: str
    # Similarity, higher is better. For unit vectors the range is [-1, 1]:
    # anti-correlated vectors score negative. Real embeddings of real text are
    # rarely negative against a real query, but the type does not promise it.
    # pgvector's `<#>` returns *negative* inner product, so the adapter negates
    # it — a sign error there returns the least similar chunks and looks
    # entirely plausible until somebody reads the results.
    score: float
    char_start: int
    char_end: int
    # How much of `text` is borrowed lead-in from the previous chunk:
    # `text[prefix_chars:] == document[char_start:char_end]`. Carried on the
    # result rather than left to the caller, because the arithmetic that
    # recovers it from the lengths is exactly the arithmetic a citation gets
    # silently wrong.
    prefix_chars: int = 0
    # What the chunker recorded about this span's origin — `{"definition":
    # "SyncSource._ingest"}` and the like. Surfaced here because a citation that
    # can name the function it came from is the difference between "somewhere in
    # sync.py" and a reference somebody can check.
    metadata: dict[str, Any] = field(default_factory=dict)
    # Attached by the search use case, which is the only layer that knows how
    # many retrievers ran. The stores leave it None: a store cannot describe a
    # fusion it was not part of.
    breakdown: ScoreBreakdown | None = None


class ShadowWorkspace(Protocol):
    """Somewhere to rebuild the derived tables that is not the live one.

    A replay that truncates and refills the live tables leaves the corpus
    unsearchable while it runs, and unrecoverable if it fails halfway. This is
    the seam that lets the rebuild happen alongside the live copy and then
    replace it in one step — or be thrown away, which is what `verify-replay`
    does with it.

    Deliberately not "a second database". The rebuild reads the event log and the
    artifacts, which live in the original, and copying those would both defeat
    the point and mean the copy could go stale mid-replay.
    """

    async def create(self) -> None:
        """Build empty derived tables in the workspace."""

    async def sessions(self) -> async_sessionmaker[AsyncSession]:
        """A session factory whose writes land in the workspace, not the live tables."""
        ...

    async def build_indexes(self) -> None:
        """Create the indexes that are cheaper to build after loading than during.

        The vector index in particular: building it on an empty table and
        maintaining it through every insert is both slower overall and yields a
        worse-connected graph than one build over finished data.
        """

    async def swap_in(self) -> None:
        """Replace the live derived tables with the workspace's, atomically."""

    async def discard(self) -> None:
        """Throw the workspace away, leaving the live tables untouched."""


class VectorStore(Protocol):
    async def search(
        self,
        vector: Sequence[float],
        *,
        k: int,
        filters: SearchFilters,
        ef_search: int | None = None,
    ) -> list[ScoredChunk]: ...

    async def search_exact(
        self, vector: Sequence[float], *, k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        """Sequential scan, bypassing the index.

        Ground truth for recall measurement, and nothing else. There is no way
        to know what the approximate index traded away without something
        exhaustive to compare it against.
        """
        ...


class LanguageModel(Protocol):
    """Generates prose. The only component here that can invent something.

    Every other part of this system returns text it retrieved: it can be
    irrelevant, badly ranked or out of date, but it cannot be *fabricated*. A
    language model asked to answer from context will smooth over a gap with a
    fluent, plausible claim drawn from its training data, and nothing in the
    response distinguishes that from a claim drawn from the corpus.

    So the port is deliberately thin — a system prompt, a user message, a
    string back — and everything that makes an answer trustworthy lives outside
    it: what goes into the context, and what is checked afterwards. Nothing here
    can be relied on to behave, which is why `domain/grounding.py` verifies the
    output rather than trusting the instruction.

    `async` unlike `Embedder` and `Reranker`, because this is a network call
    rather than local matrix work: the event loop should be free while it waits.
    """

    @property
    def model_id(self) -> str:
        """Identity of the model, recorded on every answer it produces."""
        ...

    async def complete(
        self, system: str, user: str, *, max_tokens: int = 1024
    ) -> str:
        """One completion.

        Raises `TransientError` for rate limits and timeouts, so the worker's
        existing backoff applies unchanged, and `PermanentError` for a safety
        block or an empty response — retrying either produces the same nothing,
        and an empty string presented as an answer is worse than a failure.
        """
        ...


class Reranker(Protocol):
    """A cross-encoder: reads the query and the document *together*.

    The `Embedder` above is a bi-encoder. It encodes a query and a document
    separately, which is what lets every document vector be computed once at
    ingest and searched with a vector lookup — and it is exactly what limits it,
    because the model never sees the pair. It has to compress a document into
    384 numbers without knowing what will be asked of it.

    A cross-encoder reads both at once and scores the pair directly, which is far
    more accurate and impossible to precompute: there is one forward pass per
    query-document pair, at query time. That asymmetry is the whole design.
    Retrieve fifty cheaply, score those fifty expensively, return ten.

    **It cannot recover what retrieval missed.** A document that was not in the
    shortlist cannot be ranked into it, however relevant. That is why this runs
    after fusion rather than instead of it, and why M2.1 and M2.2 came first.

    Synchronous, and deliberately so. This is CPU-bound matrix work, the same as
    embedding; making it `async` would advertise concurrency the implementation
    does not have. Callers push it to a thread.
    """

    @property
    def model_id(self) -> str:
        """Identity of the loaded model, for logging and for the breakdown."""
        ...

    @property
    def max_length(self) -> int:
        """Tokens the model reads per pair. Longer input is truncated to fit."""
        ...

    def rerank(self, query: str, documents: Sequence[str]) -> list[float]:
        """Relevance scores, one per document, in the order given.

        Higher is better. The scale is the model's own and is not comparable to
        a cosine similarity or a fused RRF score — which is why the caller
        replaces the score outright rather than combining it with anything.
        """
        ...


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    """One entity the extractor found, with offsets it has *verified*.

    `char_start` and `char_end` index into the text that was handed to
    `extract`, and the invariant is exact:

        text[char_start:char_end] == name

    That is checked by the extractor before this object exists, and it is the
    whole reason the offsets are on this type rather than left to the caller.
    A language model asked where a name appears returns a plausible number; it
    is guessing, and a mention stored at a guessed offset points at whatever
    text happens to live there. The provenance chain M2.5 built for citations is
    only worth having if every link in it was checked.

    No `canonical_name` here. Canonicalisation is the storage layer's decision
    and M3.2's problem; an extractor that returned one would be quietly voting
    on resolution.
    """

    name: str
    type: EntityType
    confidence: float
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class EntityRef:
    """An already-resolved entity, offered to the extractor as a candidate.

    Carries the id because the relationship must point at *this* entity rather
    than at a name that might be resolved differently later, and the name
    because the model can only reason about names.
    """

    entity_id: UUID
    name: str
    type: EntityType


@dataclass(frozen=True, slots=True)
class ExtractedRelationship:
    """One typed, directed claim, with the span that asserts it.

    `subject_id` and `object_id` are ids from the candidate list rather than
    names the model produced, which is what makes an invented endpoint
    impossible to store rather than merely unlikely — see `extract_relationships`.

    The span is optional where a mention's is not. A mention *is* its span, so
    one that cannot be located is worthless; a relationship's quote is
    supporting evidence, and an edge whose sentence could not be found verbatim
    is still a usable edge with weaker provenance.
    """

    subject_id: UUID
    object_id: UUID
    predicate: Predicate
    confidence: float
    evidence: str
    char_start: int | None = None
    char_end: int | None = None


class EntityExtractor(Protocol):
    """Finds the things a piece of text talks about.

    Chunk-level rather than memory-level, for two reasons that point the same
    way. A 500-token chunk gives markedly better precision than a 5,000-token
    document — the model has less room to drift into summarising — and the
    mention needs chunk-level offsets anyway, so extracting per document would
    mean mapping offsets back onto chunks afterwards.

    `version` follows the M1.4 chunker pattern and exists for the same reason:
    it encodes the model *and* the prompt, so improving extraction becomes a
    query — select the mentions carrying the old version and redo only those —
    instead of a full corpus rebuild. Two extractors that disagree must never
    share a version string, because the version is the only record of what
    produced a mention.
    """

    @property
    def version(self) -> str:
        """Identifies the extractor, its model, and its prompt."""
        ...

    async def extract(self, text: str, *, kind: MemoryKind) -> list[ExtractedEntity]:
        """Entities explicitly present in `text`, with verified offsets.

        `kind` steers the prompt rather than the parsing: code wants
        identifiers and library names, prose wants people and organisations, and
        a single prompt that tries to serve both returns English words from
        source files.
        """
        ...

    async def extract_relationships(
        self, text: str, entities: Sequence[EntityRef]
    ) -> list[ExtractedRelationship]:
        """Typed, directed claims between entities the caller already has.

        **The candidate list is the guardrail.** The model links between
        supplied entities rather than naming its own, so a relationship cannot
        point at something that does not exist — which is the failure that would
        otherwise be invisible, because an edge to a fabricated entity looks
        exactly like an edge to a real one until somebody follows it.

        The entities passed in are the *resolved* ones from M3.2. Passing
        pre-resolution entities would attach relationships to nodes that are
        about to be merged away, and every edge on a loser would have to be
        re-pointed by hand afterwards.

        Anything naming a subject or object outside the supplied set is dropped
        by the implementation, not by the caller. The model does occasionally
        invent one.
        """
        ...


@dataclass(frozen=True, slots=True)
class MemoryNode:
    """A memory as the graph knows it: identity and nothing else.

    Four fields, and the omissions are the design. No content, no chunks, no
    normalized text — those live in Postgres, which stays the system of record.
    A graph carrying its own copy would be a second source of truth, and the
    question "which one is right?" has no good answer once they differ.

    What is here is what a traversal needs to *report* what it found: an id to
    join back on, a key a person recognises, and a kind and a date to filter by
    before paying for the join.
    """

    memory_id: UUID
    external_key: str
    kind: MemoryKind
    occurred_at: datetime | None


@dataclass(frozen=True, slots=True)
class EntityNode:
    """A thing the corpus talks about. Populated by M3.1's extraction.

    `name` is what the text said; `canonical_name` is what M3.2 resolved it to,
    and the two are separate fields rather than one overwritten field because
    losing the surface form loses the evidence for the resolution. "Dr. Chen",
    "Chen" and "chen@example.com" collapsing to one canonical entity is only
    reviewable if what each mention actually said survives.

    `confidence` is the extractor's, and it is on the node rather than the edge
    because it describes how sure we are the entity *exists*, not how sure we
    are that a given memory mentions it.
    """

    entity_id: UUID
    name: str
    canonical_name: str
    type: str
    confidence: float


@dataclass(frozen=True, slots=True)
class SourceNode:
    """The connector a memory came from.

    Declared by M3.0's schema — label, constraint and `FROM_SOURCE` edge type —
    and written by nothing until M3.4. That gap was not harmless: `link` merges
    its endpoints, so a `FROM_SOURCE` edge written without this would have
    created a `Source` node carrying an id and no name, which is the one failure
    the port's own docstring warns about.

    Projected because it is what makes the graph's memories filterable the way
    every other read of the corpus is. `(source_name, external_key)` is the
    durable identity of an item everywhere else in this system; without the
    source, the graph knows only half of it.
    """

    source_id: UUID
    name: str
    kind: str


@dataclass(frozen=True, slots=True)
class GraphNode:
    """A node by label and identity, with whatever properties it carries.

    The untyped counterpart to `MemoryNode` and `EntityNode` above: those are
    what a writer supplies, this is what a *reader* gets back, because a
    traversal returns whatever it walked through and cannot know in advance
    which labels that will be.
    """

    label: GraphLabel
    key: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    """One relationship, with both endpoints named by identity.

    The endpoints are `GraphNode` rather than ids alone because a relationship
    needs to know which label it is attaching to: `entity_id` and `memory_id`
    are both UUIDs, and a Cypher `MATCH` against the wrong label matches
    nothing at all rather than failing.
    """

    type: EdgeType
    start: GraphNode
    end: GraphNode
    properties: dict[str, Any] = field(default_factory=dict)
    # Property names that are part of the relationship's *identity*, and therefore
    # belong inside the `MERGE` pattern rather than in the `SET` after it.
    #
    # Empty for `MENTIONS` and `FROM_SOURCE`, where the endpoints are the whole
    # identity. Not empty for `RELATES_TO`, and that is not a refinement: Neo4j
    # merges one relationship per (type, start, end), so two predicates between one
    # pair — "sqlalchemy uses postgres" and "sqlalchemy depends_on postgres" — were
    # collapsing into a single edge whose predicate was whichever one was written
    # last. Both claims are in the corpus, the projection reported 25 edges, the
    # graph held 24, and which of the two survived depended on row order.
    #
    # Only names in `domain.values.EDGE_IDENTITY_PROPERTIES` are accepted, because
    # these are interpolated into Cypher: a property name cannot be a bound
    # parameter any more than a label can.
    identity: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GraphPath:
    """A walk from the queried node to something else, in order.

    `nodes` and `edges` interleave: `nodes[i]` is joined to `nodes[i + 1]` by
    `edges[i]`, so there is always exactly one fewer edge than node. The whole
    path is returned rather than just the far end because the *route* is the
    answer — "what connects this decision to that person" is not answered by
    naming the person.
    """

    nodes: tuple[GraphNode, ...]
    edges: tuple[EdgeType, ...]

    @property
    def length(self) -> int:
        """Hops. A path to an immediate neighbour has length 1."""
        return len(self.edges)


@dataclass(frozen=True, slots=True)
class GraphReach:
    """One memory the graph reached from a seed entity, and the route that reached it.

    `route` is the entity names along the path, seed first, and it is not
    decoration: it is what `ScoreBreakdown.graph_path` carries, and the reason a
    graph contribution can be argued with rather than merely observed. A result
    promoted because it shares an entity with something you already found is
    explicable; one promoted by "the graph said so" is not.

    `hops` is the *graph* path length, not the entity-hop depth. Two entities
    connected by a `RELATES_TO` edge are one hop apart; two co-mentioned in the
    same memory are two, because the path runs through the memory. Both are one
    entity hop, and the distinction is kept here rather than flattened because a
    typed relationship is a stronger claim than a co-mention and the ranking is
    entitled to say so.
    """

    memory_id: UUID
    # Which chunk of that memory named the entity. An ordinal, not an id, because
    # that is what the projection stores — see `graph_projection._mention_edges`.
    chunk_ordinal: int
    entity_id: UUID
    hops: int
    route: tuple[str, ...]


class GraphStore(Protocol):
    """The graph projection: relationships as first-class objects.

    **Why this is not more Postgres tables.** Fixed one- and two-hop joins are
    genuinely fine in SQL and frequently faster; a join table with the right
    index beats a graph database at "which entities does this memory mention".
    The break comes at *variable-depth* traversal — "what connects this decision
    to that person?" is two hops or five, and which one is not known when the
    query is written. In SQL that is a recursive CTE whose cost and readability
    both degrade with depth; in Cypher it is `[*1..5]`. Traversal also costs
    what the neighbourhood costs rather than what the tables cost, because the
    relationships are stored as pointers rather than resolved through an index
    on every hop.

    **Everything here is a projection, and `clear` is the proof.** Nothing in
    this store is source of truth: it is rebuilt from Postgres, and on any
    disagreement Postgres wins and the graph is thrown away. That is only
    affordable because Phase 1 designed the derived state to be disposable —
    and it is why no use case may write here directly. A use case that wrote to
    the graph and to Postgres would have made the graph a second source of
    truth in everything but name, and the first replay would silently discard
    whatever only the graph knew.

    Every write is idempotent (`MERGE`, never `CREATE`), because a rebuild that
    could not be run twice would not be a rebuild.
    """

    async def upsert_memory(self, node: MemoryNode) -> None: ...

    async def upsert_entity(self, node: EntityNode) -> None: ...

    async def upsert_source(self, node: SourceNode) -> None: ...

    async def link(self, edge: GraphEdge) -> None: ...

    async def prune_memories(self, memory_ids: Sequence[UUID]) -> int:
        """Detach-delete these `Memory` nodes. Returns how many were there.

        **The one place a node-level delete is legitimate, and it is legitimate
        because of what calls it.** A use case that kept the graph current by
        deleting the parts it changed would be maintaining a second source of
        truth by hand — the thing this port's docstring forbids. What calls this
        is the projection sync, whose entire job is to make a *neighbourhood* of
        the graph equal to what Postgres says it should be, and which does it the
        same way the full rebuild does: delete, then re-project. Without a scoped
        delete, "incremental" would mean "additive", and a mention that Postgres
        no longer has would survive in the graph until the next full rebuild.

        Detaching is what makes the re-projection a rebuild rather than a merge:
        every `MENTIONS` and `FROM_SOURCE` edge this memory had goes with it, so
        an edge that should no longer exist cannot survive by not being
        mentioned.
        """
        ...

    async def prune_entities(self, entity_ids: Sequence[UUID]) -> int:
        """Detach-delete these `Entity` nodes. Returns how many were there.

        The counterpart for resolution: a merged-away entity has to *leave* the
        graph, and there is no upsert that expresses "no longer exists". Callers
        must re-sync every memory that mentioned one, because detaching takes
        those `MENTIONS` edges with it — `memories_mentioning` exists so that set
        can be found before the delete rather than guessed at afterwards.
        """
        ...

    async def mention_edges(
        self,
        *,
        memory_ids: Sequence[UUID] = (),
        entity_ids: Sequence[UUID] = (),
    ) -> list[tuple[UUID, UUID]]:
        """`(memory_id, entity_id)` for every `MENTIONS` edge touching either set.

        Read from the graph rather than from Postgres deliberately, and the
        difference is the whole point of the call: the sync needs to know what the
        *graph* currently claims, because the rows that produced those claims may
        already have moved or gone.

        Both directions, because the sync needs both and each covers a case the
        other cannot see. An entity that has just lost its last mention is
        unreachable from Postgres — nothing there associates it with any memory —
        but the graph still has the edge, and without it the entity's node survives
        every scoped sync as an orphan. After a merge it is the mirror image: the
        loser is unreachable from Postgres, and only the graph can say which
        memories have to be re-projected once its node is pruned.
        """
        ...

    async def neighbours(
        self, entity_id: UUID, *, depth: int = 2, limit: int = 50
    ) -> list[GraphPath]:
        """Paths of 1 to `depth` hops out from an entity, shortest first.

        Undirected: `MENTIONS` points from a memory to an entity, so a
        direction-respecting traversal from an entity would find nothing at all.

        `limit` bounds paths rather than nodes, and the two differ sharply — a
        well-connected entity has combinatorially many paths to the same handful
        of neighbours. Shortest-first is what makes the bound useful rather than
        arbitrary.
        """
        ...

    async def reach(
        self,
        seed_entity_ids: Sequence[UUID],
        *,
        depth: int = 2,
        exclude_entity_ids: Sequence[UUID] = (),
        limit: int = 200,
    ) -> list[GraphReach]:
        """Memories reachable from these entities, shortest route first.

        **This is the read M3.5 exists to make, and the one that justifies a graph
        database rather than more Postgres tables.** It is a variable-depth
        traversal whose bound is a parameter: `depth=1` is a join, `depth=2` is a
        join of a join, and the recursive CTE that expresses the family degrades in
        both cost and readability with every level. Here it is `[*1..4]`.

        `depth` counts *entity* hops, and the query bounds graph hops at twice that
        — an entity reaches another either directly, through a `RELATES_TO` edge, or
        through a memory that mentions both. Depth 2 is the default because depth 3
        on a connected graph reaches most of the corpus, which is not a ranking.

        `exclude_entity_ids` is hub suppression, and it is applied *inside* the
        traversal rather than to its results. An entity mentioned in a third of the
        corpus does not merely add noise at the end: it is a bridge every path can
        cross, so at depth 2 it connects everything to everything. Excluding hubs
        afterwards would leave the paths that ran through them.

        Undirected, for the reason `neighbours` is: `MENTIONS` points from a memory
        to an entity, so a direction-respecting walk from an entity finds nothing.

        `limit` bounds rows, and the ordering is what makes the bound meaningful:
        shortest route first, then by id so a rerun agrees.
        """
        ...

    async def clear(self) -> None:
        """Empty the projection, leaving the schema in place."""
        ...

    async def all_nodes(self) -> list[GraphNode]:
        """Every projected node, with its properties. For divergence detection.

        Reads the whole graph, and a sampled version of this would not be worth
        having: the question it answers is "does the projection differ from
        Postgres anywhere", and a check that looks at some of the nodes answers a
        different question. `SchemaVersion` is excluded, because it describes the
        schema rather than the projection.
        """
        ...

    async def all_edges(self) -> list[GraphEdge]:
        """Every projected relationship, endpoints named by identity.

        The endpoints carry label and key only. Their properties are `all_nodes`'
        business, and duplicating them per edge would make an edge diff report a
        node's changed property once per relationship it happens to have.
        """
        ...


class KeywordStore(Protocol):
    """Lexical retrieval, and the deliberate opposite of `VectorStore`.

    They fail in complementary ways, which is the entire argument for having
    both. An embedding of `SKIP LOCKED` carries almost no meaning, so the vector
    half scores that query 0.000 on this corpus; a term index finds it
    immediately, because the term is rare and appears in one place. Run it the
    other way — "microservices" against a document that says "service-oriented
    split" — and the term index scores nothing at all.

    **The return type is `ScoredChunk`, the same shape `VectorStore` returns.**
    That is the whole reason this protocol is shaped like the one above: M2.2
    fuses the two rankings, and a fusion that first has to translate between two
    result types is a fusion with a place for the two halves to disagree about
    what a chunk is.

    Scores are *not* comparable across the two: this returns `ts_rank_cd`
    output, unbounded and corpus-relative, while the vector store returns a
    similarity in [-1, 1]. Anything combining them has to rank-normalise first,
    which is exactly what RRF does and why M2.2 is a separate milestone.
    """

    async def search(
        self, query: str, *, k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        """The top k chunks matching `query`, best first.

        Returns an empty list — never raises — when the query reduces to no
        lexemes, which is what "the of and" does. A query that finds nothing is
        an ordinary answer, not an error.
        """
        ...
