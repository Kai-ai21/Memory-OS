"""Ports: what the application layer needs from the outside world.

Protocols only. No implementations, no imports from `memoryos.adapters`. The
dependency arrow points inward: adapters implement these, never the reverse.
"""

from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol
from uuid import UUID

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

    async def append(self, event: IngestionEvent) -> None: ...

    async def replay(self, after_seq: int = 0, limit: int = 1000) -> Sequence[IngestionEvent]: ...


class MemoryRepository(Protocol):
    async def get_current(self, source_id: UUID, external_key: str) -> Memory | None: ...

    async def add_version(self, memory: Memory) -> None: ...

    async def tombstone(self, memory_id: UUID) -> None: ...


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
    """A retrievable span, with offsets into the document it came from."""

    ordinal: int
    text: str
    char_start: int
    char_end: int
    token_count: int


class Chunker(Protocol):
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
