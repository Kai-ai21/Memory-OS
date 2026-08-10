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
from memoryos.domain.values import ContentHash, MemoryKind, SourceKind, TimeProvenance


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

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "fused": self.fused,
            "vector_rank": self.vector_rank,
            "keyword_rank": self.keyword_rank,
            "vector_score": self.vector_score,
            "keyword_score": self.keyword_score,
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
