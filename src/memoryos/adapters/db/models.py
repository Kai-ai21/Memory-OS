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
    # Replay order. BY DEFAULT rather than ALWAYS so a replay or an import can
    # supply its own sequence when reconstructing the log.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=False), nullable=False)
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
        CheckConstraint("ordinal >= 0", name="ck_memory_chunks_ordinal_non_negative"),
        CheckConstraint("token_count > 0", name="ck_memory_chunks_token_count_positive"),
        CheckConstraint("char_start >= 0", name="ck_memory_chunks_char_start_non_negative"),
        CheckConstraint("char_end > char_start", name="ck_memory_chunks_char_range"),
        Index("ix_memory_chunks_chunker_version", "chunker_version"),
        Index("ix_memory_chunks_content_hash", "content_hash"),
    )
