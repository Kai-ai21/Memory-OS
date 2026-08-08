"""Domain entities.

Frozen dataclasses that mirror the persisted schema without knowing that SQL
exists. Use cases work with these; adapters translate them (see
`memoryos.adapters.db.mappers`).

Fields the database assigns — identity sequences and `server_default` timestamps
— are typed `| None` and are populated on read. A `None` there means "not yet
persisted", not "absent".
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from memoryos.domain.values import (
    ContentHash,
    EventType,
    MemoryKind,
    SourceKind,
    TimeProvenance,
)


@dataclass(frozen=True, slots=True)
class Source:
    """An origin that artifacts are ingested from."""

    id: UUID
    kind: SourceKind
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    # Opaque and source-owned: only the connector that wrote it may interpret it.
    cursor: dict[str, Any] = field(default_factory=dict)
    # Incremental sync cannot observe deletions — a vanished file produces no
    # event. Only a full sweep that reconciles the observed set against the known
    # set can find them, so the two sync kinds are tracked separately.
    last_sync_at: datetime | None = None
    last_full_sync_at: datetime | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("source name must not be empty")


@dataclass(frozen=True, slots=True)
class RawArtifact:
    """Bytes observed at a source, identified by their content.

    The bytes themselves are not held here; the blob store arrives in M1.3.
    """

    content_hash: ContentHash
    byte_size: int
    media_type: str | None = None
    first_seen_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.byte_size < 0:
            raise ValueError(f"byte_size must be >= 0, got {self.byte_size}")


@dataclass(frozen=True, slots=True)
class IngestionEvent:
    """An immutable record of something observed at a source.

    The event log is the source of truth for ingestion; `Memory` and
    `MemoryChunk` are projections that can be dropped and rebuilt from it.
    """

    id: UUID
    event_type: EventType
    source_id: UUID
    # Path or identifier within the source. Not identity — content_hash is.
    external_key: str
    occurred_at_source: TimeProvenance
    # Assigned by the log on append; defines replay order.
    seq: int | None = None
    # Replay outlives the shape of the payload it replays.
    schema_version: int = 1
    # Null for deletion events, which reference no content.
    content_hash: ContentHash | None = None
    occurred_at: datetime | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    recorded_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.external_key:
            raise ValueError("external_key must not be empty")
        if self.schema_version < 1:
            raise ValueError(f"schema_version must be >= 1, got {self.schema_version}")


@dataclass(frozen=True, slots=True)
class Memory:
    """One version of one item, projected from the event log."""

    id: UUID
    source_id: UUID
    external_key: str
    content_hash: ContentHash
    kind: MemoryKind
    occurred_at_source: TimeProvenance
    version: int = 1
    is_current: bool = True
    normalized_hash: ContentHash | None = None
    title: str | None = None
    # Normalized text; populated in M1.4.
    content: str | None = None
    # When the thing happened in the world. Distinct from ingested_at, which is
    # when this system learned about it. An email from 2023 read today has both.
    occurred_at: datetime | None = None
    ingested_at: datetime | None = None
    # A nullable affordance, deliberately never computed here. A placeholder
    # heuristic becomes load-bearing the moment anything downstream trusts it.
    importance: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.external_key:
            raise ValueError("external_key must not be empty")
        if self.version < 1:
            raise ValueError(f"version must be >= 1, got {self.version}")
        # Substituting ingested_at for a missing occurred_at would collapse every
        # backfilled memory onto the day it was ingested. The unknown case gets
        # its own provenance value instead, and the two must agree in both
        # directions.
        if (self.occurred_at is None) != (self.occurred_at_source is TimeProvenance.UNKNOWN):
            raise ValueError(
                "occurred_at and occurred_at_source disagree: "
                f"occurred_at={self.occurred_at!r}, "
                f"occurred_at_source={self.occurred_at_source!r}. "
                "A null occurred_at requires TimeProvenance.UNKNOWN, and vice versa."
            )
        if self.importance is not None and not (0.0 <= self.importance <= 1.0):
            raise ValueError(f"importance must be within 0.0..1.0, got {self.importance}")


@dataclass(frozen=True, slots=True)
class MemoryChunk:
    """A retrievable span of one memory's text."""

    id: UUID
    memory_id: UUID
    # Position within the memory. Lets a hit be widened to its neighbours.
    ordinal: int
    content: str
    token_count: int
    # Offsets into the memory's normalized text, so a citation can highlight the
    # exact matched span rather than the whole document.
    char_start: int
    char_end: int
    # Versioned on the chunk, not the memory, so re-chunking can be selective.
    chunker_version: str
    content_hash: ContentHash
    embedding: tuple[float, ...] | None = None
    embedding_model: str | None = None
    embedded_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.ordinal < 0:
            raise ValueError(f"ordinal must be >= 0, got {self.ordinal}")
        if self.token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {self.token_count}")
        if self.char_start < 0:
            raise ValueError(f"char_start must be >= 0, got {self.char_start}")
        if self.char_end <= self.char_start:
            raise ValueError(
                f"char_end must be > char_start, got {self.char_end} <= {self.char_start}"
            )
