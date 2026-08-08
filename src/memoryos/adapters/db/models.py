"""SQLAlchemy models.

The persistence shape of the domain. Constraints live here rather than only in
`memoryos.domain`, because Python invariants protect this process and CHECK
constraints protect the data against every other writer.

Constraints are named explicitly so that migrations, reflection, and error
messages all refer to them the same way.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    REAL,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    MetaData,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memoryos.domain.jobs import DEFAULT_MAX_ATTEMPTS, JobStatus
from memoryos.domain.values import (
    HEX64_PATTERN,
    EventType,
    MemoryKind,
    SourceKind,
    TimeProvenance,
)

# The embedding model chosen in M1.5 (all-MiniLM-L6-v2 class) produces 384
# dimensions. The column is fixed-width, so this is not a free parameter later.
EMBEDDING_DIMENSIONS = 384

_UUID = Uuid(as_uuid=True)
_TIMESTAMPTZ = DateTime(timezone=True)
_EMPTY_JSONB = text("'{}'::jsonb")


def _enum_check(column: str, enum: type[StrEnum], name: str) -> CheckConstraint:
    """CHECK that `column` holds one of `enum`'s values.

    Generated from the enum so the database cannot fall out of step with the
    Python definition without the migration diff showing it.
    """
    allowed = ", ".join(f"'{member.value}'" for member in enum)
    return CheckConstraint(f"{column} IN ({allowed})", name=name)


class Base(DeclarativeBase):
    # Every constraint below is named explicitly; the convention only covers
    # primary keys, which SQLAlchemy otherwise leaves to Postgres to name.
    metadata = MetaData(naming_convention={"pk": "pk_%(table_name)s"})


class Source(Base):
    """An origin that artifacts are ingested from."""

    __tablename__ = "sources"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    # Opaque sync state, written and interpreted only by the owning connector.
    cursor: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    # Two timestamps because an incremental sync cannot observe a deletion: a
    # file that is gone produces no event. Only a full sweep that reconciles the
    # observed set against the known set finds them.
    last_sync_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    last_full_sync_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    created_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("kind", "name", name="uq_sources_kind_name"),
        _enum_check("kind", SourceKind, "ck_sources_kind"),
    )


class RawArtifact(Base):
    """Bytes observed at a source, addressed by their content.

    The hash is the primary key: identity is a pure function of content, so
    re-ingesting an unchanged file collides here and does nothing. That is what
    makes ingestion idempotent.

    The bytes are deliberately not stored; the blob store arrives in M1.3.
    """

    __tablename__ = "raw_artifacts"

    content_hash: Mapped[str] = mapped_column(Text, primary_key=True)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    media_type: Mapped[str | None] = mapped_column(Text)
    first_seen_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            f"content_hash ~ '{HEX64_PATTERN}'", name="ck_raw_artifacts_content_hash_hex"
        ),
        CheckConstraint("byte_size >= 0", name="ck_raw_artifacts_byte_size_non_negative"),
    )


class IngestionEvent(Base):
    """Append-only log of everything observed at a source.

    This is the source of truth for ingestion. `memories` and `memory_chunks`
    are projections that can be truncated and rebuilt from this table, which is
    what M1.7's replay proves. Rows are never updated or deleted.
    """

    __tablename__ = "ingestion_events"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    # Replay order, and the database's alone to assign. Replay reads these
    # events to rebuild projections; it never re-inserts them, so nothing has a
    # legitimate reason to supply its own sequence.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True), nullable=False)
    # Today's events will still be replayed after the payload shape changes.
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    source_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey("sources.id", name="fk_ingestion_events_source_id"),
        nullable=False,
    )
    # Path or identifier within the source. Deliberately not the event's
    # identity: replay must be deterministic, and paths move.
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    # Null for deletion events, which reference no content.
    content_hash: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("raw_artifacts.content_hash", name="fk_ingestion_events_content_hash"),
    )
    occurred_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    occurred_at_source: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    recorded_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint("seq", name="uq_ingestion_events_seq"),
        _enum_check("event_type", EventType, "ck_ingestion_events_event_type"),
        _enum_check(
            "occurred_at_source", TimeProvenance, "ck_ingestion_events_occurred_at_source"
        ),
        # The same pairing `memories` enforces. The system never fabricates a
        # timestamp anywhere. A deletion event carrying 'unknown' is correct: we
        # know when we noticed the absence (recorded_at), not when it happened.
        CheckConstraint(
            "(occurred_at IS NULL) = (occurred_at_source = 'unknown')",
            name="ck_ingestion_events_occurred_at_provenance",
        ),
        Index("ix_ingestion_events_source_id_external_key", "source_id", "external_key"),
        Index("ix_ingestion_events_recorded_at", "recorded_at"),
    )


class Memory(Base):
    """One version of one item, projected from the event log."""

    __tablename__ = "memories"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    source_id: Mapped[UUID] = mapped_column(
        _UUID, ForeignKey("sources.id", name="fk_memories_source_id"), nullable=False
    )
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))
    is_current: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("true")
    )
    content_hash: Mapped[str] = mapped_column(
        Text,
        ForeignKey("raw_artifacts.content_hash", name="fk_memories_content_hash"),
        nullable=False,
    )
    # Hash of the normalized text; set in M1.4.
    normalized_hash: Mapped[str | None] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    # Normalized text; populated in M1.4.
    content: Mapped[str | None] = mapped_column(Text)
    # When it happened in the world, versus when this system learned about it.
    # An email from 2023 ingested today has both, and they are not recoverable
    # after the fact once a source moves or changes.
    occurred_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    occurred_at_source: Mapped[str] = mapped_column(Text, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    # Nullable affordance only. Nothing computes or defaults this: a placeholder
    # heuristic becomes load-bearing as soon as anything downstream reads it.
    importance: Mapped[float | None] = mapped_column(REAL)
    # `metadata` is reserved on declarative classes, so the attribute is `meta`
    # and the column name is given explicitly.
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    deleted_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        UniqueConstraint(
            "source_id", "external_key", "version", name="uq_memories_source_key_version"
        ),
        CheckConstraint("version >= 1", name="ck_memories_version_positive"),
        CheckConstraint("importance BETWEEN 0.0 AND 1.0", name="ck_memories_importance_range"),
        # Enforced here as well as in the entity, so that a missing occurred_at
        # can never be quietly backfilled with the ingestion timestamp by any
        # writer, including psql.
        CheckConstraint(
            "(occurred_at IS NULL) = (occurred_at_source = 'unknown')",
            name="ck_memories_occurred_at_provenance",
        ),
        _enum_check("kind", MemoryKind, "ck_memories_kind"),
        _enum_check("occurred_at_source", TimeProvenance, "ck_memories_occurred_at_source"),
        # Exactly one current version per item.
        Index(
            "uq_memories_current_version",
            "source_id",
            "external_key",
            unique=True,
            postgresql_where=text("is_current"),
        ),
        Index("ix_memories_occurred_at", "occurred_at"),
        Index("ix_memories_ingested_at", "ingested_at"),
    )


class MemoryChunk(Base):
    """A retrievable span of one memory's text."""

    __tablename__ = "memory_chunks"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    # CASCADE is required by the deletion guardrail. Without it a deleted
    # memory's chunks stay in the vector index and keep surfacing in results.
    memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey("memories.id", name="fk_memory_chunks_memory_id", ondelete="CASCADE"),
        nullable=False,
    )
    # Position within the memory, so a hit can be widened to its neighbours.
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Offsets into the memory's text, so a citation can highlight the matched
    # span rather than the whole document.
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    # Versioned per chunk rather than per memory, which permits re-chunking a
    # subset without rewriting everything.
    chunker_version: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # No HNSW index here. That is M1.6, and it is built after bulk loading
    # rather than maintained incrementally through every insert.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    embedding_model: Mapped[str | None] = mapped_column(Text)
    embedded_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        UniqueConstraint("memory_id", "ordinal", name="uq_memory_chunks_memory_ordinal"),
        CheckConstraint(
            f"content_hash ~ '{HEX64_PATTERN}'", name="ck_memory_chunks_content_hash_hex"
        ),
        CheckConstraint("ordinal >= 0", name="ck_memory_chunks_ordinal_non_negative"),
        CheckConstraint("token_count > 0", name="ck_memory_chunks_token_count_positive"),
        CheckConstraint("char_start >= 0", name="ck_memory_chunks_char_start_non_negative"),
        CheckConstraint("char_end > char_start", name="ck_memory_chunks_char_range"),
        Index("ix_memory_chunks_chunker_version", "chunker_version"),
        Index("ix_memory_chunks_content_hash", "content_hash"),
    )


class Job(Base):
    """Durable work queue.

    A table rather than a broker, for two reasons that are not convenience.

    Enqueueing a job and writing the data it refers to happen in one
    transaction, so there is no window in which one committed and the other did
    not. With a broker that window is real, and the standard fix — the
    transactional outbox — is a jobs table in the database anyway.

    And `SELECT status, count(*) FROM jobs GROUP BY 1` is the whole monitoring
    story. Every failure's error and traceback are queryable with SQL.

    The ceiling is low thousands of jobs per second. Embedding throughput will
    be orders of magnitude under that, and if Phase 6 ever needs more, the
    `JobQueue` port swaps out without a use case noticing.
    """

    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    job_type: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    # Collapses duplicate enqueues of the same logical work. Null opts out.
    dedupe_key: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{JobStatus.PENDING.value}'")
    )
    # Higher runs first.
    priority: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    # Incremented when a job is claimed, not when it fails. A worker that
    # segfaults never reaches its failure handler, so counting on failure would
    # let a job that reliably kills the process retry forever with attempts
    # stuck at zero. Counting on claim guarantees every job exhausts its budget.
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text(str(DEFAULT_MAX_ATTEMPTS))
    )
    run_after: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    locked_by: Mapped[str | None] = mapped_column(Text)
    locked_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    lease_expires_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    last_error: Mapped[str | None] = mapped_column(Text)
    last_traceback: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        _enum_check("status", JobStatus, "ck_jobs_status"),
        CheckConstraint("attempts >= 0", name="ck_jobs_attempts_non_negative"),
        CheckConstraint("max_attempts >= 1", name="ck_jobs_max_attempts_positive"),
        # A running job without a lease can never be reclaimed: the sweeper
        # finds expired leases, and a null one never expires.
        CheckConstraint(
            "status <> 'running' OR (locked_by IS NOT NULL AND lease_expires_at IS NOT NULL)",
            name="ck_jobs_running_requires_lease",
        ),
    )


# Declared outside the class so the claim index can use a real column object for
# `priority DESC`; `__table_args__` runs before the attributes are bound.

# The claim index. Its columns and predicate mirror the claim query's ORDER BY
# and WHERE exactly, which is the only way the planner can satisfy the ordering
# from the index instead of sorting. Partial, because in a mature queue almost
# every row is 'succeeded' and indexing those would be permanently growing dead
# weight.
Index(
    "ix_jobs_claim",
    Job.priority.desc(),
    Job.run_after,
    postgresql_where=text("status = 'pending'"),
)

# Idempotent enqueue: the same logical work cannot be queued twice while it is
# still in flight. Once it finishes, the key is free again.
Index(
    "uq_jobs_dedupe",
    Job.job_type,
    Job.dedupe_key,
    unique=True,
    postgresql_where=text("dedupe_key IS NOT NULL AND status IN ('pending', 'running')"),
)

# The sweeper's index: find running jobs whose lease has expired.
Index("ix_jobs_lease", Job.lease_expires_at, postgresql_where=text("status = 'running'"))

# Observability. `SELECT status, count(*) FROM jobs GROUP BY 1` and its friends.
Index("ix_jobs_status_type", Job.status, Job.job_type)


class EmbeddingCacheEntry(Base):
    """Text-to-vector memoisation, keyed by (model, text).

    A table of its own rather than a column on `memory_chunks`, because
    identical text in different memories should be embedded once. M1.4's smoke
    test found five identical empty files in this repository alone.

    The model identity is inside the key, and that is a correctness
    requirement rather than an optimisation: without it, upgrading the model
    would silently reuse vectors from the old one. Nothing would error — the
    index would simply hold two incompatible coordinate systems, and
    similarity between them is arithmetically valid and semantically
    meaningless. That is a very hard failure to trace back to its cause.
    """

    __tablename__ = "embedding_cache"

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSIONS), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint(
            f"cache_key ~ '{HEX64_PATTERN}'", name="ck_embedding_cache_key_hex"
        ),
        CheckConstraint("dimension > 0", name="ck_embedding_cache_dimension_positive"),
    )


# Model lookups during a re-embed scan by model rather than by key.
Index("ix_embedding_cache_model", EmbeddingCacheEntry.model_id)
