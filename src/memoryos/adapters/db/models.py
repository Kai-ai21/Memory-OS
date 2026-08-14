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
    Computed,
    DateTime,
    Float,
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
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from memoryos.domain.events import EventKind
from memoryos.domain.jobs import DEFAULT_MAX_ATTEMPTS, JobStatus
from memoryos.domain.surfacing import SurfaceReason
from memoryos.domain.values import (
    HEX64_PATTERN,
    AssumptionVerdict,
    ConfidenceHorizon,
    DecisionStatus,
    EntityType,
    EventType,
    EvidenceKind,
    EvidenceRelation,
    MemoryKind,
    MergeStatus,
    MergeStrategy,
    OutcomeVerdict,
    PatternKind,
    PatternRelation,
    Predicate,
    SourceKind,
    SuggestionStatus,
    TimeProvenance,
    Verdict,
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
    # Which extractor last ran over this memory, whatever it found.
    #
    # M3.1 keyed the skip on "does this memory have mentions at the current
    # version", which is a different question and answers it wrongly for the
    # memories that legitimately contain no entities. Those write no rows, so
    # they never satisfy the check, so every run re-extracts them — for real
    # money, forever, and the pending count never reaches zero. Measured: 56
    # memories processed, 34 with mentions, and the queue barely moved.
    #
    # Recording the attempt rather than inferring it from its output is the
    # only thing that distinguishes "not yet done" from "done, found nothing".
    entity_extractor_version: Mapped[str | None] = mapped_column(Text)
    # The same marker for M3.3, and separate from the entity one so the two
    # prompts can be improved and re-run independently.
    relationship_extractor_version: Mapped[str | None] = mapped_column(Text)

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
    # span rather than the whole document. They bound the chunk's own span, not
    # `content`: the invariant is `content[prefix_chars:] ==
    # memory.content[char_start:char_end]`.
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    # How much of `content` is overlap borrowed from the preceding chunk. A
    # column rather than arithmetic on the lengths, because a derived value
    # every reader has to rediscover is one most readers get wrong — the UI had
    # to measure the corpus to work out what the offsets meant.
    prefix_chars: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    # Versioned per chunk rather than per memory, which permits re-chunking a
    # subset without rewriting everything.
    chunker_version: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    # What the chunker knew about where this span came from — the enclosing
    # definition's name, for code. Persisted rather than held in memory, because
    # the moment a citation needs it is query time, and until M1.7 it was
    # computed during normalization and then discarded. Same `metadata`/`meta`
    # split as `Memory`, for the same declarative-attribute reason.
    meta: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    # The HNSW index over this column arrives in migration 0005, declared
    # below the class. It is built after the pipeline that fills the column
    # rather than maintained through every insert from an empty table.
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    embedding_model: Mapped[str | None] = mapped_column(Text)
    embedded_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    # The lexical half of retrieval, derived from `content` by Postgres itself.
    # Declared here rather than only in the migration so the shadow schema — which
    # is copied from this metadata — gets it too, and so `alembic check` has
    # something to compare against.
    #
    # Never read in Python. It is a query-side structure: `keyword_store.py`
    # matches against it with `@@` and ranks with `ts_rank_cd`, and nothing loads
    # it into a mapped instance. A generated column cannot be written either, so
    # the pipeline neither knows nor needs to know it exists.
    search_vector: Mapped[str | None] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    __table_args__ = (
        UniqueConstraint("memory_id", "ordinal", name="uq_memory_chunks_memory_ordinal"),
        CheckConstraint(
            f"content_hash ~ '{HEX64_PATTERN}'", name="ck_memory_chunks_content_hash_hex"
        ),
        CheckConstraint("ordinal >= 0", name="ck_memory_chunks_ordinal_non_negative"),
        CheckConstraint("token_count > 0", name="ck_memory_chunks_token_count_positive"),
        CheckConstraint("char_start >= 0", name="ck_memory_chunks_char_start_non_negative"),
        CheckConstraint("char_end > char_start", name="ck_memory_chunks_char_range"),
        CheckConstraint(
            "prefix_chars >= 0", name="ck_memory_chunks_prefix_chars_non_negative"
        ),
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


class QueryJudgement(Base):
    """A human's verdict on one result for one query.

    The only table in this schema that a machine cannot regenerate. Every other
    row here is either bytes observed at a source or something computed from
    them; this is somebody's opinion, and if it is lost it can only be recreated
    by asking them again. That is why `application/replay.py` classifies it
    `USER_AUTHORED` rather than squeezing it into derived or source-of-truth —
    replay must never truncate it and must never try to rebuild it.

    The query is stored as text rather than as a foreign key to some `queries`
    table. A judgement is *about* a phrasing: "how does claiming work" and "how
    does the job queue claim work" are different queries with different right
    answers, and normalising them into one row would erase the distinction the
    golden set exists to measure.

    **The identity of the judged item is its natural key, not a memory id, and
    there is deliberately no foreign key to `memories`.** That is a departure from
    the milestone's stated schema and it is forced, because the two requirements
    it gives are contradictory: a replay deletes and recreates every memory row
    with a new UUID, so any reference to `memories.id` either blocks the replay
    or dies with it. Measured on this database — `TRUNCATE memory_chunks,
    memories, jobs CASCADE` reports "truncate cascades to table
    judgement_probe" and empties it, and a plain `DELETE FROM memories` takes it
    too via `ON DELETE CASCADE`. Every full replay would silently destroy the
    golden set, which is precisely what `USER_AUTHORED` exists to prevent.

    So `(source_name, external_key)` is the identity — the same natural key
    `verify-replay` compares on, and for the same reason: ids are minted per
    write and a rebuild legitimately changes them. `memory_id` and `chunk_id`
    survive as *snapshots*, plain columns recording what the system pointed at
    when the human judged, alongside `rank_at_judgement` and
    `score_at_judgement` which are snapshots for the same reason. The export
    re-resolves the natural key to whatever is current, so the golden set stays
    correct across any number of rebuilds.

    **`chunk_ordinal` extends that identity downwards.** NULL means the verdict
    is about the memory; a number means it is about that chunk of it, and only
    that chunk counts as a hit. It is part of the key rather than a snapshot
    because, unlike `chunk_id`, an ordinal is stable across a rebuild — chunking
    is deterministic, so chunk 4 of a file is chunk 4 again after a replay.
    """

    __tablename__ = "query_judgements"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    # The durable identity of the judged item, stable across rebuilds.
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    # What the system pointed at when the verdict was given. No foreign key: see
    # the class docstring. Null once a rebuild has moved on, which is honest —
    # the judgement is still valid, the pointer simply is not.
    memory_id: Mapped[UUID | None] = mapped_column(_UUID)
    chunk_id: Mapped[UUID | None] = mapped_column(_UUID)
    # Which chunk the verdict is about, or NULL for "this memory, whichever chunk
    # matched". Part of the identity rather than a snapshot like `chunk_id`,
    # because the ordinal survives a rebuild and the id does not — and because a
    # memory-level verdict cannot express the failure that motivated this column:
    # the right file returned on the wrong chunk.
    chunk_ordinal: Mapped[int | None] = mapped_column(Integer)
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    rank_at_judgement: Mapped[int | None] = mapped_column(Integer)
    score_at_judgement: Mapped[float | None] = mapped_column(REAL)
    # The filters in force when the judgement was made. A result judged
    # irrelevant under a source filter is a different statement from the same
    # verdict over the whole corpus.
    filters: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    judged_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # One verdict per (query, item). Re-judging updates rather than
        # appending, so the golden set cannot hold two contradictory opinions
        # about the same pair and quietly average them. Keyed on the natural
        # identity rather than `memory_id`, which would let the same judgement be
        # recorded again under a new id after every rebuild.
        #
        # `nulls_not_distinct` is what keeps that true now that the ordinal is in
        # the key. Under Postgres' default, NULLs are distinct in a unique index,
        # so every memory-level row — the overwhelming majority — would stop
        # colliding with itself and the upsert would append instead of replace.
        UniqueConstraint(
            "query_text",
            "source_name",
            "external_key",
            "chunk_ordinal",
            name="uq_query_judgements_query_item",
            postgresql_nulls_not_distinct=True,
        ),
        _enum_check("verdict", Verdict, "ck_query_judgements_verdict"),
        CheckConstraint(
            "rank_at_judgement IS NULL OR rank_at_judgement >= 1",
            name="ck_query_judgements_rank_positive",
        ),
        CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_query_judgements_chunk_ordinal_non_negative",
        ),
        CheckConstraint("length(btrim(query_text)) > 0", name="ck_query_judgements_query_text"),
        # A `missing` verdict has no rank, because the point of it is that the
        # result was not in the ranking at all. Enforced rather than trusted:
        # a rank on a missing row would silently corrupt any recall computed
        # from this table.
        CheckConstraint(
            "verdict <> 'missing' OR rank_at_judgement IS NULL",
            name="ck_query_judgements_missing_has_no_rank",
        ),
        Index("ix_query_judgements_query_text", "query_text"),
    )


# Model lookups during a re-embed scan by model rather than by key.
Index("ix_embedding_cache_model", EmbeddingCacheEntry.model_id)

# Approximate nearest neighbour over chunk embeddings. `vector_ip_ops` because
# the embedder normalizes to unit length, which makes inner product and cosine
# similarity the same number and inner product the cheaper one. `m` and
# `ef_construction` are fixed at build time; `hnsw.ef_search` is the knob the
# adapter sets per query.
Index(
    "ix_memory_chunks_embedding_hnsw",
    MemoryChunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_ip_ops"},
)

# The lexical index, and the counterpart to the one above. GIN rather than GiST:
# slower to build, much faster to query, and this is written once per chunk and
# read on every keyword search. Unlike the HNSW index it is *not* deferred during
# a shadow rebuild — an inverted index has no graph connectivity to degrade from
# being grown incrementally, so there is nothing to gain by building it late.
Index(
    "ix_memory_chunks_search",
    MemoryChunk.search_vector,
    postgresql_using="gin",
)


class Entity(Base):
    """A thing the corpus talks about, as Postgres knows it.

    **Postgres is the system of record; the Neo4j `Entity` node is a
    projection.** The row is written first and committed, and the graph is
    updated afterwards — so a crash between the two leaves a corpus that is
    correct and a graph that is behind, which the next extraction repairs. The
    other order would leave a graph asserting entities no query can join to.

    `canonical_name` is a *minimal* normalisation — casefold and collapsed
    whitespace, nothing more — and identity here is `(canonical_name, type)`.
    That is exact-match deduplication, not resolution: it knows "Neo4j" and
    "neo4j" are one entity and has no opinion whatsoever about "Dr. Chen" versus
    "Chen", or "Postgres" versus "PostgreSQL". Doing more would be M3.2's job
    done early and badly, and it would shrink the duplicate count M3.2 is
    scoped against — improving the number by moving the ruler.

    Without the unique constraint the table has no identity at all: every
    re-extraction of the same chunk would insert another row for the same name,
    and "the twenty most-mentioned entities" would be a list of twenty
    coincidences.

    `confidence` is the extractor's confidence that the entity *exists*, kept on
    the entity rather than only on the mention because it describes the thing,
    not the sighting. Per-sighting confidence lives on `entity_mentions`.
    """

    __tablename__ = "entities"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    # The surface form as first seen. Kept alongside the canonical form because
    # losing what the text actually said loses the evidence for any later
    # resolution decision.
    name: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_name: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(REAL)
    first_seen_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    # M3.2. Non-null means this entity was merged away: its mentions now belong
    # to the winner, and it survives only so the merge can be undone. Every read
    # that counts or traverses entities must exclude these, and `_upsert_entity`
    # must follow the pointer — otherwise the next extraction re-attaches
    # mentions to a merged-away row and silently undoes the resolution.
    #
    # `ON DELETE SET NULL` rather than CASCADE: deleting a winner must not
    # delete the entities that were merged into it. They become active again,
    # which is wrong-but-recoverable, where cascading would be data loss.
    merged_into_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "entities.id",
            name="fk_entities_merged_into_id",
            ondelete="SET NULL",
        ),
    )

    __table_args__ = (
        UniqueConstraint("canonical_name", "type", name="uq_entities_canonical_type"),
        # An entity merged into itself is unreachable through any read that
        # follows the pointer, and would loop one that followed it repeatedly.
        CheckConstraint(
            "merged_into_id IS NULL OR merged_into_id <> id",
            name="ck_entities_not_merged_into_self",
        ),
        _enum_check("type", EntityType, "ck_entities_type"),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_entities_confidence_range",
        ),
        CheckConstraint("length(btrim(name)) > 0", name="ck_entities_name_non_empty"),
        CheckConstraint(
            "length(btrim(canonical_name)) > 0", name="ck_entities_canonical_non_empty"
        ),
    )


class EntityMention(Base):
    """One place in one chunk where an entity was named.

    **The offsets are the point of this table.** `char_start` and `char_end`
    index into `memory_chunks.content`, exactly, so an entity leads back to the
    span of text that produced it — the same provenance chain M2.5 built for
    citations, and worth exactly as much as its weakest link. They are written
    only after the extractor has confirmed that the name really appears there;
    an offset a language model reported is a guess, and a mention stored at a
    guessed offset points at whatever text happens to occupy it.

    `UNIQUE (entity_id, chunk_id, char_start)` is what makes re-extraction
    idempotent at the row level, and the columns are exactly the natural key: the
    same entity named twice in one chunk is two mentions, at two offsets, and
    both are real.

    `extractor_version` follows the M1.4 chunker-version pattern. It encodes the
    model and the prompt, so improving extraction is a query — find the mentions
    carrying the old version, redo those — rather than a corpus-wide rebuild.

    Both foreign keys cascade. A deleted memory takes its chunks, and its chunks
    take their mentions: a mention whose chunk is gone has no text to point at,
    and keeping it would leave the provenance chain ending in nothing.
    """

    __tablename__ = "entity_mentions"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    entity_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "entities.id",
            name="fk_entity_mentions_entity_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memories.id",
            name="fk_entity_mentions_memory_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    chunk_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memory_chunks.id",
            name="fk_entity_mentions_chunk_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float | None] = mapped_column(REAL)
    extractor_version: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "entity_id", "chunk_id", "char_start", name="uq_entity_mentions_span"
        ),
        CheckConstraint(
            "char_start >= 0", name="ck_entity_mentions_char_start_non_negative"
        ),
        CheckConstraint("char_end > char_start", name="ck_entity_mentions_char_range"),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_entity_mentions_confidence_range",
        ),
    )


# The skip check runs per memory on every extraction job: "are there already
# mentions for this memory at the current extractor version?". Without this it
# is a sequential scan of the mentions table on every job.
Index(
    "ix_entity_mentions_memory_version",
    EntityMention.memory_id,
    EntityMention.extractor_version,
)

# "The twenty most-mentioned entities", and every later traversal that starts
# from an entity and asks where it was seen.
Index("ix_entity_mentions_entity", EntityMention.entity_id)

# Resolution's read pattern in M3.2: find the candidates a name might collapse
# into. Also what makes the duplicate measurement cheap.
Index("ix_entities_canonical_name", Entity.canonical_name)

# Every read of *active* entities filters on this column, which after a
# resolution run is most reads in the system.
Index("ix_entities_merged_into", Entity.merged_into_id)

# The extraction queue's predicate: memories not yet extracted at this version.
Index("ix_memories_entity_extractor_version", Memory.entity_extractor_version)


class EntityMerge(Base):
    """One resolution decision: two entities are the same thing, or might be.

    **Both a ledger and a review queue.** `status` distinguishes the two: a
    `pending` row is a proposal nobody has acted on, `applied` is a merge in
    force, `reverted` is one that was undone. One table rather than two because
    a pending candidate and an applied merge carry identical information and
    differ only in whether somebody said yes — and splitting them would mean
    moving rows between tables to answer that question.

    **Nothing is deleted, ever.** The losing entity keeps its row and gains a
    `merged_into_id`; its mentions are repointed at the winner. Resolution is
    never perfect — this milestone's own report names the merges it got wrong —
    so a merge that cannot be undone is a corpus that degrades every time the
    resolver is run and improves never.

    `moved_mention_ids` is what makes `unmerge` exact rather than approximate.
    Repointing is destructive: once `entity_mentions.entity_id` says "winner",
    nothing records which of the winner's mentions used to be the loser's, and
    an unmerge would have to guess. Recording the ids at merge time is the
    difference between restoring the previous state and restoring something that
    resembles it.

    `evidence` is not decoration either. A reviewer looking at the pending queue
    is being asked to make a judgement, and "0.87" is not something a person can
    judge — "cosine 0.87 between 'postgres' and 'postgresql'" is.
    """

    __tablename__ = "entity_merges"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    winner_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "entities.id", name="fk_entity_merges_winner_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    loser_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "entities.id", name="fk_entity_merges_loser_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{MergeStatus.PENDING.value}'")
    )
    confidence: Mapped[float] = mapped_column(REAL, nullable=False)
    evidence: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    # The mentions this merge moved, so an unmerge puts back exactly those.
    moved_mention_ids: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    proposed_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    merged_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    reverted_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        _enum_check("strategy", MergeStrategy, "ck_entity_merges_strategy"),
        _enum_check("status", MergeStatus, "ck_entity_merges_status"),
        CheckConstraint(
            "confidence BETWEEN 0.0 AND 1.0", name="ck_entity_merges_confidence_range"
        ),
        # An entity cannot be merged into itself. Reachable through the manual
        # command with the same id twice, and it would repoint every mention to
        # where it already is and then mark the entity merged into itself —
        # unrecoverable without a manual UPDATE.
        CheckConstraint("winner_id <> loser_id", name="ck_entity_merges_distinct"),
        # An applied merge has a timestamp; a pending one does not. Keeps the
        # two representations of "is this in force" from disagreeing.
        CheckConstraint(
            "(status = 'applied' AND merged_at IS NOT NULL) "
            "OR (status = 'pending' AND merged_at IS NULL) "
            "OR (status = 'reverted' AND merged_at IS NOT NULL "
            "    AND reverted_at IS NOT NULL)",
            name="ck_entity_merges_status_timestamps",
        ),
    )


# One entity can be merged away exactly once at a time. Partial, because a
# reverted merge legitimately leaves the pair free to be merged again — and
# without the predicate, undoing a merge would permanently forbid redoing it.
Index(
    "uq_entity_merges_active_loser",
    EntityMerge.loser_id,
    unique=True,
    postgresql_where=text("status = 'applied'"),
)

# The same pair is not proposed twice while a proposal is outstanding. A
# re-run of the resolver must not grow the review queue by a copy of itself.
Index(
    "uq_entity_merges_pending_pair",
    EntityMerge.winner_id,
    EntityMerge.loser_id,
    unique=True,
    postgresql_where=text("status = 'pending'"),
)

# The review queue's own read pattern.
Index("ix_entity_merges_status", EntityMerge.status, EntityMerge.confidence)


class EntityRelationship(Base):
    """One typed, directed claim about two entities, and where it was made.

    **Every row carries the chunk that asserted it, and that is the design.**
    Without provenance a relationship is an unfalsifiable claim: something in
    the system believes React depends on Postgres and nothing can say why, which
    is precisely the failure Phase 2 spent M2.5 eliminating for answers. A
    Phase 3 answer built on these edges has to stay as citable as a Phase 2 one,
    and it can only be as citable as its weakest edge.

    **The same relationship asserted in five chunks is five rows.** That is not
    duplication to be collapsed — it is the evidence, and M3.5 weights edges by
    how often the corpus repeats them. One assertion is a claim; five is a
    pattern. Collapsing them at write time would throw away the only signal that
    distinguishes the two, and the unique constraint is scoped per chunk for
    exactly that reason.

    Direction is carried by the column names rather than by convention:
    `subject_id` does `predicate` to `object_id`, and nothing anywhere is
    allowed to read it the other way.
    """

    __tablename__ = "entity_relationships"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    subject_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "entities.id", name="fk_entity_relationships_subject_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    object_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "entities.id", name="fk_entity_relationships_object_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    predicate: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(REAL)
    # Where the claim was made. Both, because a memory locates it for a reader
    # and a chunk locates it for a citation.
    memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memories.id", name="fk_entity_relationships_memory_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    chunk_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memory_chunks.id",
            name="fk_entity_relationships_chunk_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    # The span that asserts it, into `memory_chunks.content`. Nullable, unlike a
    # mention's offsets: the model quotes the sentence rather than pointing at a
    # name, and a quote that cannot be located is worth keeping without a span
    # where a *mention* without one would be worthless.
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    extractor_version: Mapped[str] = mapped_column(Text, nullable=False)
    extracted_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # Per chunk, so repetition across chunks survives as separate rows.
        UniqueConstraint(
            "subject_id",
            "predicate",
            "object_id",
            "chunk_id",
            name="uq_entity_relationships_assertion",
        ),
        _enum_check("predicate", Predicate, "ck_entity_relationships_predicate"),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_entity_relationships_confidence_range",
        ),
        # A self-relationship is never informative and is what a model returns
        # when it has nothing to say about a chunk that names one entity.
        CheckConstraint(
            "subject_id <> object_id", name="ck_entity_relationships_distinct"
        ),
        CheckConstraint(
            "(char_start IS NULL) = (char_end IS NULL)",
            name="ck_entity_relationships_span_pairing",
        ),
        CheckConstraint(
            "char_start IS NULL OR char_start >= 0",
            name="ck_entity_relationships_char_start_non_negative",
        ),
        CheckConstraint(
            "char_end IS NULL OR char_end > char_start",
            name="ck_entity_relationships_char_range",
        ),
    )


# The traversal read pattern: everything this entity does, and everything done
# to it. Two indexes because direction means the two questions are different.
Index(
    "ix_entity_relationships_subject",
    EntityRelationship.subject_id,
    EntityRelationship.predicate,
)
Index(
    "ix_entity_relationships_object",
    EntityRelationship.object_id,
    EntityRelationship.predicate,
)

# The skip check, and the per-memory delete that replaces a version's output.
Index(
    "ix_entity_relationships_memory_version",
    EntityRelationship.memory_id,
    EntityRelationship.extractor_version,
)


class ChangeSummary(Base):
    """A generated description of what changed between two versions.

    **The one cache here that cannot go stale.** Both memories are rows nothing
    ever updates, so the diff between them is fixed and so is any description of
    it. Everywhere else in this system a cache is a bet that the input has not
    moved; here the input provably cannot.

    Keyed on the pair rather than on the newer version, because the useful diff
    is not always against the immediate predecessor — "what changed between v1
    and v4" is a different question from three consecutive diffs, and both are
    worth keeping.

    `summarizer_version` is part of the key, following the M1.4 chunker and M3.1
    extractor pattern. Improving the prompt then becomes a query — find the rows
    at the old version, redo those — rather than emptying a table that mostly
    contains summaries nobody has complained about.

    `grounded` and `unsupported_terms` are stored as they were found, not
    recomputed on read. The check ran against the diff *the model was shown*, and
    a reader recomputing it against a diff rebuilt with different context
    settings would get a different verdict for the same stored text.
    """

    __tablename__ = "change_summaries"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    from_memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memories.id", name="fk_change_summaries_from_memory_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    to_memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memories.id", name="fk_change_summaries_to_memory_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    summarizer_version: Mapped[str] = mapped_column(Text, nullable=False)
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    grounded: Mapped[bool] = mapped_column(Boolean, nullable=False)
    unsupported_terms: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "from_memory_id",
            "to_memory_id",
            "summarizer_version",
            name="uq_change_summaries_pair_version",
        ),
        CheckConstraint(
            "from_memory_id <> to_memory_id", name="ck_change_summaries_distinct_pair"
        ),
    )


class Decision(Base):
    """What was decided, what was considered, why, and what had to be true.

    **The second `USER_AUTHORED` table, and the first one that is the product
    rather than the measurement.** `query_judgements` records an opinion about a
    search result; this records an opinion about the world, made at a moment,
    under uncertainty. Neither is derivable from the log, and replay must
    therefore neither truncate nor rebuild this — see `application/replay.py`.
    Phase 5 reads this table for everything it does, so a row a machine invented
    here becomes a behavioural claim in M5.3 and a reflection in M5.4, both of
    which sound insightful and neither of which can be falsified. That is why
    the extraction path writes to `decision_suggestions` and never here.

    **No foreign key to `memories`, deliberately, and the lesson is M2.0a's.**
    A decision is *about* something, and the memories that informed it are
    linked through `decision_evidence`; the decision itself references nothing
    that a rebuild recreates. Any column here pointing at `memories.id` would
    put this table inside `TRUNCATE memories CASCADE` and every routine replay
    would delete the corpus Phase 5 operates on.

    `confidence` is confidence **at the time of deciding**, and it is never
    refreshed. Its whole value is that it was recorded before the outcome was
    known: a number updated in hindsight measures nothing, because everyone is
    well calibrated about the past.

    **`confidence_horizon` is the column that makes that sentence checkable, and
    Phase 5 shipped without it.** Immutability guarantees the number did not
    *move*; it guarantees nothing about when it was first written. A confidence
    reconstructed a week later is still never refreshed, and the calibration
    table built on it looks exactly like one built on real foresight — which is
    what M5.3 measured and could not say. Set once at capture from
    `domain/patterns.classify_confidence` and never updated, because a horizon
    somebody can revise is a horizon somebody can revise into whichever answer
    makes the table look better. The CHECK pairs it with `confidence` so a null
    number and a claimed horizon cannot coexist.

    `decided_at_source` follows M1.1 exactly, and for the same reason. A date a
    person typed is `declared`; a date read off an ADR's front matter is
    `parsed`; a date taken from a file's mtime is `filesystem`. Phase 4's
    weighting rules apply unchanged, which is what stops a decision dated by a
    file's modification time from being ranked as though somebody had asserted
    it. Unlike `memories.occurred_at` this column is NOT NULL — a decision with
    no date is not a decision anybody can reason about later — so there is no
    `unknown` pairing to enforce, and the CHECK forbids `unknown` outright.
    """

    __tablename__ = "decisions"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    chosen: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(REAL)
    confidence_horizon: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{ConfidenceHorizon.UNKNOWN.value}'")
    )
    expected_outcome: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(_TIMESTAMPTZ, nullable=False)
    decided_at_source: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{DecisionStatus.OPEN.value}'")
    )
    created_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        _enum_check("status", DecisionStatus, "ck_decisions_status"),
        _enum_check("decided_at_source", TimeProvenance, "ck_decisions_decided_at_source"),
        # `decided_at` is NOT NULL, so the M1.1 pairing collapses into a
        # prohibition: there is no null date for `unknown` to describe, and a
        # row claiming unknown provenance for a date it has would be asserting
        # two contradictory things about the same column.
        CheckConstraint(
            f"decided_at_source <> '{TimeProvenance.UNKNOWN.value}'",
            name="ck_decisions_decided_at_known",
        ),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_decisions_confidence_range",
        ),
        _enum_check("confidence_horizon", ConfidenceHorizon, "ck_decisions_horizon"),
        # A confidence with no horizon is a number nobody can calibrate against,
        # and a horizon with no confidence is a claim about a number that is not
        # there. Neither may exist.
        CheckConstraint(
            f"(confidence IS NULL) = "
            f"(confidence_horizon = '{ConfidenceHorizon.UNKNOWN.value}')",
            name="ck_decisions_horizon_pairing",
        ),
        CheckConstraint("length(btrim(question)) > 0", name="ck_decisions_question"),
        CheckConstraint("length(btrim(chosen)) > 0", name="ck_decisions_chosen"),
        Index("ix_decisions_status_decided_at", "status", "decided_at"),
    )


class DecisionOption(Base):
    """One thing that was on the table, chosen or not.

    **The rejected rows are the reason this table exists.** A record with one
    option is a description of what happened; the alternatives and why each lost
    are what make it a decision, and `application/decisions.py` refuses to write
    a decision that has none. That rule lives in the use case rather than here
    because it is a statement about a *set* of rows: a CHECK sees one row at a
    time, and the options are inserted in the same transaction as the decision
    they belong to, so any constraint strong enough to catch the empty case
    would also reject the moment before the first option is written.

    `rejected_because` is separate from the decision's `reasoning` and they are
    not two names for one field. The reasoning says why the winner won;
    `rejected_because` says why *this particular* alternative lost, and the two
    diverge constantly — an option can be rejected for a reason that has nothing
    to do with what made the winner attractive.
    """

    __tablename__ = "decision_options"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    decision_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id", name="fk_decision_options_decision_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    was_chosen: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    rejected_because: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint(
            "length(btrim(description)) > 0", name="ck_decision_options_description"
        ),
        # An option that was taken cannot also carry a reason it was not taken.
        # Reachable through an edit that flips `was_chosen` and leaves the old
        # rejection text behind, which would then read as a rejection of the
        # thing that was chosen.
        CheckConstraint(
            "NOT was_chosen OR rejected_because IS NULL",
            name="ck_decision_options_chosen_has_no_rejection",
        ),
    )


# Exactly one option per decision may be the chosen one. Partial, because the
# constraint is about the chosen row rather than about the rejected ones — there
# is no limit on how many alternatives were considered.
Index(
    "uq_decision_options_one_chosen",
    DecisionOption.decision_id,
    unique=True,
    postgresql_where=text("was_chosen"),
)

Index("ix_decision_options_decision", DecisionOption.decision_id)


class DecisionAssumption(Base):
    """Something that had to be true for the choice to be the right one.

    **The load-bearing table of Phase 5.** An outcome says a decision worked or
    it did not, which is one bit about one decision and generalises to nothing.
    An assumption says *why*, and assumptions repeat across decisions that have
    nothing else in common. "Deployment will take two days" failing six times is
    a pattern with a name and a fix; six unrelated bad projects is noise with a
    mood.

    It is also the field a person will not volunteer. Everyone can say what they
    chose and most people can say why; almost nobody lists what they were taking
    for granted unless asked, one at a time, which is what `decide --interactive`
    does and why that prompt is doing real work rather than filling a form.

    `held` and `evaluated_at` are written by M5.2 and are null until then. They
    are declared now, unpopulated, for the reason `occurred_at` was declared in
    M1.1: the column is cheap today and the *history* it would have recorded is
    unrecoverable later. A NULL `held` means "not yet judged" and is
    deliberately distinct from `false` — a system that could not tell an
    unevaluated assumption from a broken one would report every new decision as
    built on sand.
    """

    __tablename__ = "decision_assumptions"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    decision_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id",
            name="fk_decision_assumptions_decision_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(REAL)
    # M5.2 writes all three, or none of them.
    #
    # Widened from `BOOLEAN` in migration 0018. Forcing a binary produced noise
    # rather than data: almost nothing anybody assumes is cleanly right or
    # wrong, and `partially` is the answer for the ones that were true until the
    # week they were not. The column keeps its M5.0 name — `held = 'failed'`
    # reads oddly and renaming it to fix one sentence is how a schema and its
    # documentation drift apart.
    held: Mapped[str | None] = mapped_column(Text)
    evaluated_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    # Why the evaluator reached that verdict. Separate from the statement, which
    # is what was believed at the time and must never be edited to match what
    # happened — an assumption rewritten in hindsight is a decision record
    # arguing with itself.
    note: Mapped[str | None] = mapped_column(Text)
    # M5.2's grouping. Null means ungrouped, which is the common case: an
    # assumption nothing else in the corpus resembles is not a failure of the
    # grouper, it is a belief held once.
    #
    # `ON DELETE SET NULL` rather than CASCADE: deleting a group must ungroup
    # its members, never delete the assumptions themselves.
    group_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "assumption_groups.id",
            name="fk_decision_assumptions_group_id",
            ondelete="SET NULL",
        ),
    )

    __table_args__ = (
        _enum_check("held", AssumptionVerdict, "ck_decision_assumptions_held"),
        CheckConstraint(
            "length(btrim(statement)) > 0", name="ck_decision_assumptions_statement"
        ),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_decision_assumptions_confidence_range",
        ),
        # A verdict without a date cannot be placed in time, and a date without a
        # verdict claims an evaluation that produced nothing. The pairing is the
        # same shape as M1.1's `occurred_at`/`occurred_at_source` rule and exists
        # for the same reason: it stops a later writer filling in half of it.
        CheckConstraint(
            "(held IS NULL) = (evaluated_at IS NULL)",
            name="ck_decision_assumptions_evaluation_pairing",
        ),
    )


Index("ix_decision_assumptions_decision", DecisionAssumption.decision_id)

# The stats query's read pattern: how many assumptions hold, grouped.
Index("ix_decision_assumptions_held", DecisionAssumption.held)
Index("ix_decision_assumptions_group", DecisionAssumption.group_id)


class AssumptionGroup(Base):
    """Assumptions from different decisions that say the same thing.

    **This is what makes M5.3 possible.** A pattern is the same assumption
    failing repeatedly, and "the same assumption" is not a string comparison:
    "this will take two days", "the deploy is straightforward" and "integration
    should be quick" are one recurring belief wearing three sentences. Without
    grouping, every one of them is a sample of size one and no pattern can exist.

    `label` is the statement of whichever member the group was built around,
    kept as a human-readable handle rather than as a canonical form. There is
    deliberately no attempt to synthesise a better label from the members: a
    generated summary of three sentences is a fourth sentence nobody wrote, and
    M5.3 would then be finding patterns in text this module invented.

    The same asymmetry M3.2 states drives the thresholds: **a false grouping is
    worse than a missed one.** A missed group leaves two beliefs looking
    unrelated, which is visible and fixable by accepting a pending candidate. A
    false group invents a recurrence — four members, one hold rate, a finding
    about how somebody estimates — out of assumptions that have nothing to do
    with each other.
    """

    __tablename__ = "assumption_groups"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    # Whether a person put this group together or the embedder did. The same
    # distinction `EvidenceKind` makes for outcomes, and needed for the same
    # reason: an auto-grouped set at 0.94 cosine and a hand-curated one are not
    # equally trustworthy inputs to a claim about somebody's judgement.
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        _enum_check("strategy", MergeStrategy, "ck_assumption_groups_strategy"),
        CheckConstraint("length(btrim(label)) > 0", name="ck_assumption_groups_label"),
    )


class AssumptionGroupCandidate(Base):
    """Two assumptions the embedder thinks might be the same belief.

    The review queue, and M3.2's `entity_merges` in miniature: a pair scoring
    between the review floor and the auto threshold is information — "we looked
    at these two and were not sure" — and a system that discarded it would
    re-propose the same pair on every run forever.

    Keyed on the *pair* rather than on a group, because at the first run there
    are no groups to propose membership of. Accepting merges both sides into one
    group, creating it if neither has one, which is the only formulation that
    works whether zero, one or both are already grouped.

    Rejections are kept for the reason M5.0's are: the pair is then excluded
    from later runs, and the count of rejections is the only measurement of how
    often the embedder proposes two beliefs that are not the same.
    """

    __tablename__ = "assumption_group_candidates"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    left_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decision_assumptions.id",
            name="fk_assumption_group_candidates_left_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    right_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decision_assumptions.id",
            name="fk_assumption_group_candidates_right_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    similarity: Mapped[float] = mapped_column(REAL, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{MergeStatus.PENDING.value}'")
    )
    # The embedder that scored it. Part of the row for the reason M3.1's
    # extractor version is: a different model produces different numbers, and a
    # queue mixing two of them is a queue with no threshold.
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    proposed_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        _enum_check("status", MergeStatus, "ck_assumption_group_candidates_status"),
        CheckConstraint(
            "similarity BETWEEN 0.0 AND 1.0",
            name="ck_assumption_group_candidates_similarity_range",
        ),
        # An assumption is not a candidate to be grouped with itself.
        CheckConstraint(
            "left_id <> right_id", name="ck_assumption_group_candidates_distinct"
        ),
        CheckConstraint(
            "(status = 'pending') = (reviewed_at IS NULL)",
            name="ck_assumption_group_candidates_review_pairing",
        ),
    )


# The same pair is not proposed twice while a proposal is outstanding. Partial
# for the `entity_merges` reason: a rejected pair legitimately becomes
# proposable again under a different embedder.
Index(
    "uq_assumption_group_candidates_pending_pair",
    AssumptionGroupCandidate.left_id,
    AssumptionGroupCandidate.right_id,
    unique=True,
    postgresql_where=text("status = 'pending'"),
)

Index(
    "ix_assumption_group_candidates_status",
    AssumptionGroupCandidate.status,
    AssumptionGroupCandidate.similarity,
)


class AssumptionEvidence(Base):
    """A memory that bears on whether an assumption held.

    The third table with this exact shape — `decision_evidence`,
    `outcome_evidence`, and now this one — and the third for the same reasons.
    The foreign keys cascade, so a memory leaving the corpus takes its links and
    leaves the evaluation standing; the natural key beside the ids is what a
    replay re-links against, because `TRUNCATE memories CASCADE` reaches this
    table whatever `application/replay.py` classifies it as. It is listed in
    `EVIDENCE_TABLES`, and a test derives that list from this metadata so that
    forgetting the fourth fails the build.

    **The evidence is attached by the person doing the evaluating, not by the
    thing that proposed it.** `assumptions suggest` retrieves memories that bear
    on an assumption and prints them; nothing here is written until somebody
    evaluates the assumption and names which of them they actually used. An
    evidence row is therefore a claim a human made, which is what M5.4 needs it
    to be.
    """

    __tablename__ = "assumption_evidence"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    assumption_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decision_assumptions.id",
            name="fk_assumption_evidence_assumption_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memories.id", name="fk_assumption_evidence_memory_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    chunk_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "memory_chunks.id",
            name="fk_assumption_evidence_chunk_id",
            ondelete="CASCADE",
        ),
    )
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_ordinal: Mapped[int | None] = mapped_column(Integer)
    occurred_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    linked_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "assumption_id",
            "memory_id",
            "chunk_id",
            name="uq_assumption_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_assumption_evidence_chunk_ordinal_non_negative",
        ),
        CheckConstraint(
            "(chunk_id IS NULL) = (chunk_ordinal IS NULL)",
            name="ck_assumption_evidence_chunk_pairing",
        ),
        Index("ix_assumption_evidence_assumption", "assumption_id"),
        Index("ix_assumption_evidence_memory", "memory_id"),
    )


class DecisionEvidence(Base):
    """A memory that informed a decision, records it, or argues against it.

    **Both keys, and each does a different job.** `memory_id` and `chunk_id` are
    real foreign keys with `ON DELETE CASCADE`, so a memory that leaves the
    corpus takes its evidence rows with it — a link to a document that no longer
    exists is a citation to nothing, and this system spent M2.5 making sure a
    citation always resolves. The decision itself is untouched by that, which is
    the point: a decision survives losing a piece of its evidence.

    `source_name`, `external_key` and `chunk_ordinal` are the same natural key
    `query_judgements` uses, and they are here because the cascade above has a
    consequence a replay makes routine. A full rebuild truncates `memories`, so
    `TRUNCATE ... CASCADE` takes every row in this table with it — exactly the
    trap M1.7 found for the golden set. The fix there was to key on something
    that survives a rebuild, and that is what these three columns are: replay
    snapshots this table before it truncates and re-links it afterwards by
    natural key, so evidence outlives a rebuild even though the row itself does
    not. See `ReplayCorpus._preserve_evidence`.

    `chunk_ordinal` is part of the durable key for the reason it is part of a
    judgement's: chunking is deterministic, so chunk 4 of a file is chunk 4
    again after a replay, while `chunk_id` is minted per write.
    """

    __tablename__ = "decision_evidence"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    decision_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id", name="fk_decision_evidence_decision_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memories.id", name="fk_decision_evidence_memory_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    # Null links the whole memory; a value links the one chunk that carries the
    # passage. M2.5's provenance, which is what a suggestion arrives with.
    chunk_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "memory_chunks.id",
            name="fk_decision_evidence_chunk_id",
            ondelete="CASCADE",
        ),
    )
    # The durable identity of the same thing, for re-linking after a rebuild.
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_ordinal: Mapped[int | None] = mapped_column(Integer)
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    linked_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # One statement per (decision, item, relation). The same memory may
        # legitimately both inform a decision and record it — two rows, two
        # relations — but asserting the same relation twice is a duplicate, not
        # extra evidence. `nulls_not_distinct` for the M2.0a reason: without it
        # every memory-level link, which is most of them, stops colliding with
        # itself.
        UniqueConstraint(
            "decision_id",
            "memory_id",
            "chunk_id",
            "relation",
            name="uq_decision_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        _enum_check("relation", EvidenceRelation, "ck_decision_evidence_relation"),
        CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_decision_evidence_chunk_ordinal_non_negative",
        ),
        # A chunk-level link needs the ordinal that survives a rebuild, or the
        # re-link after a replay would silently widen it to the whole memory.
        CheckConstraint(
            "(chunk_id IS NULL) = (chunk_ordinal IS NULL)",
            name="ck_decision_evidence_chunk_pairing",
        ),
        Index("ix_decision_evidence_decision", "decision_id"),
        Index("ix_decision_evidence_memory", "memory_id"),
    )


class DecisionSuggestion(Base):
    """A draft decision record an extractor proposed, waiting to be judged.

    **Nothing here is a decision until a person says so.** The queue is the
    whole safety property of the extraction path: a language model asked to find
    decisions in a corpus of explanatory prose will find them, because prose
    that explains a choice looks exactly like a record of one, and the fabricated
    remainder — a confidence nobody held, an assumption nobody made — is
    indistinguishable from the real thing once it is a row. Every later phase
    reads `decisions`, so a fabricated record there produces a behavioural claim
    that is both plausible and unfalsifiable.

    The draft is stored as JSONB rather than as a half-populated decision with a
    status. A pending suggestion is not a decision in an early state — it is a
    model's reading of a passage, and giving it a row in `decisions` would mean
    every query in Phase 5 had to remember to exclude it. One forgotten `WHERE`
    is then a pattern built on drafts.

    **Provenance is a natural key plus snapshots, and there is no foreign key to
    `memories`.** This table is `USER_AUTHORED` — it carries somebody's accept
    or reject — so it must survive the replay that its provenance points into,
    which is only possible if the pointer is not a constraint. `memory_id` and
    `chunk_id` are what the system pointed at when the suggestion was made, and
    go null-shaped rather than dangling after a rebuild; `(source_name,
    external_key, chunk_ordinal)` is what still resolves.

    Rejections are kept. They are the only measurement of what the extractor
    gets wrong, and a queue that deleted them would re-propose the same passage
    on the next run and look like it had found something new.
    """

    __tablename__ = "decision_suggestions"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    # The proposal itself: question, chosen, reasoning, options, assumptions.
    # Shaped by `application/decisions.DecisionDraft`, which is what validates it
    # on the way in and on the way out.
    draft: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    # The passage the model read, stored verbatim. The review queue shows it
    # beside the draft so that accepting is a judgement about evidence rather
    # than about plausibility — a draft alone always reads well.
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_ordinal: Mapped[int | None] = mapped_column(Integer)
    memory_id: Mapped[UUID | None] = mapped_column(_UUID)
    chunk_id: Mapped[UUID | None] = mapped_column(_UUID)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{SuggestionStatus.PENDING.value}'")
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    suggester_version: Mapped[str] = mapped_column(Text, nullable=False)
    # Set when an accept wrote a decision. `ON DELETE SET NULL` rather than
    # CASCADE: deleting a decision must not erase the record that a suggestion
    # was once accepted, which is part of how the extractor is scored.
    decision_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id",
            name="fk_decision_suggestions_decision_id",
            ondelete="SET NULL",
        ),
    )
    suggested_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        _enum_check("status", SuggestionStatus, "ck_decision_suggestions_status"),
        CheckConstraint(
            "length(btrim(source_text)) > 0", name="ck_decision_suggestions_source_text"
        ),
        CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_decision_suggestions_chunk_ordinal_non_negative",
        ),
        # A reviewed suggestion has a review timestamp and a pending one does
        # not, so the two representations of "has somebody looked at this"
        # cannot disagree. The same shape as `entity_merges`.
        CheckConstraint(
            "(status = 'pending') = (reviewed_at IS NULL)",
            name="ck_decision_suggestions_review_pairing",
        ),
        # Only an accepted suggestion may name a decision. A rejected draft
        # pointing at a real record would make the extractor look right about a
        # decision somebody wrote by hand.
        CheckConstraint(
            "decision_id IS NULL OR status = 'accepted'",
            name="ck_decision_suggestions_decision_requires_accept",
        ),
    )


# The queue's own read pattern, and the ordering the review UI walks.
Index(
    "ix_decision_suggestions_status",
    DecisionSuggestion.status,
    DecisionSuggestion.suggested_at,
)

# The same passage is not proposed twice while a proposal is outstanding. Keyed
# on the durable identity rather than on `chunk_id`, so a re-run after a replay
# does not fill the queue with copies of everything already in it. Partial for
# the `entity_merges` reason: a rejected draft legitimately leaves the passage
# free to be proposed again by a better prompt.
Index(
    "uq_decision_suggestions_pending_passage",
    DecisionSuggestion.source_name,
    DecisionSuggestion.external_key,
    DecisionSuggestion.chunk_ordinal,
    unique=True,
    postgresql_where=text("status = 'pending'"),
    postgresql_nulls_not_distinct=True,
)


class DecisionOutcome(Base):
    """What actually happened after a decision.

    **The verdict and the evidence kind are two different questions and the
    second one is the load-bearing half.** `verdict` says what happened;
    `evidence_kind` says whether anybody watched it happen. A `declared` outcome
    is testimony — somebody observed the deployment, read the incident, saw the
    number move. An `inferred` one is a correlation in time plus a language
    model's opinion that the correlation means something. Both are worth
    storing and nothing downstream may average them, because the inferred kind
    is the one that scales and a pattern built mostly on it is a pattern built
    on the cheapest possible evidence.

    `too_early` is a real verdict rather than a null. Most decisions in a young
    project have no outcome yet, and "we looked and it is too soon" is a
    different fact from "nobody has looked" — the first is a corpus that has
    been maintained and the second is one with holes. It is excluded from every
    success rate, so a project with two wins and thirty unresolved decisions
    reports two out of two rather than a number that reads like a record.

    **Several outcomes per decision, deliberately, and no unique constraint on
    `decision_id`.** A decision can work in the first month and fail in the
    sixth, and collapsing that into one mutable row would destroy exactly the
    sequence M5.3 exists to find. `observed_at` orders them, and a `too_early`
    recorded early is not contradicted by a `worked` recorded later — it is the
    honest first half of the story.

    `confidence` is confidence in *this reading of what happened*, which is a
    different quantity from the decision's own confidence at the time and is
    stored on a different table for that reason. A manual outcome is 1.0 by
    construction: you observed it.

    No foreign key to `memories`. The decision does not have one either, for the
    reason `query_judgements` does not — the evidence lives in
    `outcome_evidence`, and a column here pointing at a row a replay recreates
    would put this table inside `TRUNCATE memories CASCADE`.
    """

    __tablename__ = "decision_outcomes"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    decision_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id", name="fk_decision_outcomes_decision_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(_TIMESTAMPTZ, nullable=False)
    # M1.1's provenance, on the outcome's own clock. A date somebody stated is
    # `declared`; one taken from the mtime of the memory that evidences it is
    # `filesystem`, and Phase 4's weighting applies to it unchanged.
    observed_at_source: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_kind: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float | None] = mapped_column(REAL)
    created_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        _enum_check("verdict", OutcomeVerdict, "ck_decision_outcomes_verdict"),
        _enum_check(
            "evidence_kind", EvidenceKind, "ck_decision_outcomes_evidence_kind"
        ),
        _enum_check(
            "observed_at_source", TimeProvenance, "ck_decision_outcomes_observed_at_source"
        ),
        # Same shape as `decisions.decided_at_source`: the column is NOT NULL,
        # so there is no missing date for 'unknown' to describe.
        CheckConstraint(
            f"observed_at_source <> '{TimeProvenance.UNKNOWN.value}'",
            name="ck_decision_outcomes_observed_at_known",
        ),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_decision_outcomes_confidence_range",
        ),
        CheckConstraint(
            "length(btrim(description)) > 0", name="ck_decision_outcomes_description"
        ),
        # An inferred outcome cannot claim certainty. Nothing produces one at 1.0
        # today, and the constraint is what stops a future writer from doing it
        # quietly: a model's reading of a correlation is never something anybody
        # observed, and a 1.0 here would let M5.3 weight it as testimony.
        CheckConstraint(
            f"evidence_kind <> '{EvidenceKind.INFERRED.value}' "
            f"OR confidence IS NULL OR confidence < 1.0",
            name="ck_decision_outcomes_inferred_is_not_certain",
        ),
        Index("ix_decision_outcomes_decision", "decision_id", "observed_at"),
        Index("ix_decision_outcomes_verdict", "verdict"),
    )


class OutcomeEvidence(Base):
    """A memory that shows an outcome happened.

    The same two-identity design as `decision_evidence`, for the same reasons
    and with the same consequence. `memory_id` and `chunk_id` cascade, so a
    memory leaving the corpus takes its evidence with it and leaves the outcome
    standing; `(source_name, external_key, chunk_ordinal)` is what a replay
    re-links against, because `TRUNCATE memories CASCADE` reaches this table
    whatever `application/replay.py` classifies it as.

    `occurred_at` is a *snapshot* of the evidence memory's own `occurred_at`,
    copied at link time rather than joined on read. That looks like
    denormalisation and is not: the whole claim an inferred outcome makes is
    that this memory occurred after that decision, and the gap between the two
    is the evidence. Joining for it would re-derive that number against whatever
    the corpus says today — and a re-sync that moves an mtime would silently
    change how strong a link somebody already reviewed and accepted.
    """

    __tablename__ = "outcome_evidence"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    outcome_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decision_outcomes.id",
            name="fk_outcome_evidence_outcome_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    memory_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "memories.id", name="fk_outcome_evidence_memory_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    chunk_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "memory_chunks.id",
            name="fk_outcome_evidence_chunk_id",
            ondelete="CASCADE",
        ),
    )
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_ordinal: Mapped[int | None] = mapped_column(Integer)
    # When the evidence happened, as the corpus said at link time. Nullable
    # because an undated memory can still be evidence a person points at — the
    # domain refuses to invent a date, and so does this.
    occurred_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    linked_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "outcome_id",
            "memory_id",
            "chunk_id",
            name="uq_outcome_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_outcome_evidence_chunk_ordinal_non_negative",
        ),
        CheckConstraint(
            "(chunk_id IS NULL) = (chunk_ordinal IS NULL)",
            name="ck_outcome_evidence_chunk_pairing",
        ),
        Index("ix_outcome_evidence_outcome", "outcome_id"),
        Index("ix_outcome_evidence_memory", "memory_id"),
    )


class OutcomeSuggestion(Base):
    """A candidate outcome the temporal layer found and a model judged.

    The M5.0 queue's shape, for a proposal that is easier to get wrong. A
    decision suggestion at least has a passage that either does or does not
    record a choice; this one asserts a *causal* relationship between two things
    on the strength of one occurring after the other. Post hoc ergo propter hoc
    is the oldest error there is and a language model shown two related-looking
    documents will make it every time, fluently.

    So the row stores the whole basis of the claim rather than only its
    conclusion, and the review UI shows all of it: the candidate memory, the
    gap in days, the window that admitted it, and which entities the two share.
    A reviewer looking at "0.7 — this looks like an outcome" is being asked to
    trust a number; one looking at "34 days later, shares `pgvector` and
    `hnsw`, here is the passage" is being asked a question they can answer.

    `entity_filter` records whether the entity constraint was *applied* or
    *unavailable*, and the distinction is not pedantry. A corpus where nothing
    has been extracted cannot fail the entity test, it simply cannot take it —
    and a candidate found by time alone is much weaker evidence than one that
    shares a resolved entity. Collapsing the two would mean a suggestion queue
    that silently changes meaning depending on whether anybody has run
    extraction lately, which is exactly what happened to this corpus.

    Provenance is a natural key plus id snapshots with no foreign key to
    `memories`, the same as `decision_suggestions`: this table is user-authored,
    it carries somebody's accept or reject, and it has to outlive the replay its
    provenance points into.
    """

    __tablename__ = "outcome_suggestions"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    decision_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id",
            name="fk_outcome_suggestions_decision_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    # description, verdict, confidence and the model's own rationale. Shaped by
    # `application/outcomes.OutcomeDraft`, which validates it both ways.
    draft: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    source_text: Mapped[str] = mapped_column(Text, nullable=False)
    source_name: Mapped[str] = mapped_column(Text, nullable=False)
    external_key: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_ordinal: Mapped[int | None] = mapped_column(Integer)
    memory_id: Mapped[UUID | None] = mapped_column(_UUID)
    chunk_id: Mapped[UUID | None] = mapped_column(_UUID)
    # The temporal claim, stored rather than recomputed. See `OutcomeEvidence`.
    candidate_occurred_at: Mapped[datetime] = mapped_column(_TIMESTAMPTZ, nullable=False)
    gap_days: Mapped[float] = mapped_column(REAL, nullable=False)
    # The window this candidate was admitted under, so a queue reviewed weeks
    # later can be read against the heuristic that produced it rather than
    # against whatever the default is by then.
    window_days: Mapped[float] = mapped_column(REAL, nullable=False)
    # The resolved entity names the decision's evidence and this candidate share.
    shared_entities: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # `applied` or `unavailable`. Text rather than a boolean because the third
    # state a boolean would invite — false meaning "no overlap" — is precisely
    # the conflation this column exists to prevent: no overlap is a rejection,
    # and no coverage is a missing test.
    entity_filter: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text(f"'{SuggestionStatus.PENDING.value}'")
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    suggester_version: Mapped[str] = mapped_column(Text, nullable=False)
    outcome_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "decision_outcomes.id",
            name="fk_outcome_suggestions_outcome_id",
            ondelete="SET NULL",
        ),
    )
    suggested_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        _enum_check("status", SuggestionStatus, "ck_outcome_suggestions_status"),
        CheckConstraint(
            "entity_filter IN ('applied', 'unavailable')",
            name="ck_outcome_suggestions_entity_filter",
        ),
        CheckConstraint(
            "length(btrim(source_text)) > 0", name="ck_outcome_suggestions_source_text"
        ),
        # The whole premise of the suggestion. A candidate that occurred before
        # the decision is not a weak outcome, it is not an outcome — and the
        # rule is enforced in the database as well as in the query that finds
        # them, because a negative gap reaching this table would be a causal
        # claim running backwards.
        CheckConstraint("gap_days > 0", name="ck_outcome_suggestions_gap_positive"),
        CheckConstraint("window_days > 0", name="ck_outcome_suggestions_window_positive"),
        CheckConstraint(
            "chunk_ordinal IS NULL OR chunk_ordinal >= 0",
            name="ck_outcome_suggestions_chunk_ordinal_non_negative",
        ),
        CheckConstraint(
            "(status = 'pending') = (reviewed_at IS NULL)",
            name="ck_outcome_suggestions_review_pairing",
        ),
        CheckConstraint(
            "outcome_id IS NULL OR status = 'accepted'",
            name="ck_outcome_suggestions_outcome_requires_accept",
        ),
    )


Index(
    "ix_outcome_suggestions_status",
    OutcomeSuggestion.status,
    OutcomeSuggestion.suggested_at,
)

# One pending proposal per (decision, candidate memory). Partial for the
# `entity_merges` reason: a rejected candidate legitimately leaves the pair free
# to be proposed again under a different window or a better prompt.
Index(
    "uq_outcome_suggestions_pending_pair",
    OutcomeSuggestion.decision_id,
    OutcomeSuggestion.source_name,
    OutcomeSuggestion.external_key,
    OutcomeSuggestion.chunk_ordinal,
    unique=True,
    postgresql_where=text("status = 'pending'"),
    postgresql_nulls_not_distinct=True,
)


class Pattern(Base):
    """A behavioural claim, with the evidence that has to exist for it to be one.

    **The most dangerous table in this schema.** Everything else here records
    something that happened; this records a generalisation about a person, and a
    generalisation is exactly the thing that sounds most like the product
    working when it is wrong. "You consistently underestimate deployment effort"
    is either a finding backed by five specific decisions or a horoscope — vague
    enough to feel true about anyone — and nothing in the sentence itself
    distinguishes the two.

    So `support_count` is not a decoration and neither is `pattern_evidence`.
    **A pattern that cannot cite is never written**: `application/patterns.py`
    refuses to emit a candidate below the minimum support, and the minimum is
    counted in *distinct decisions* rather than in rows, because four
    assumptions from two decisions is two observations rather than four.

    `confidence` is derivable rather than assigned — see
    `domain/patterns.pattern_confidence`, which states the formula and works
    four examples. A confidence number nobody can reproduce is decoration with a
    decimal point.

    `dismissed_at` and `dismissed_reason` make a pattern permanently rejectable.
    Detection over a corpus this small is never going to be right every time,
    and a claim about somebody's judgement that they have looked at and refused
    should stay refused rather than reappear on the next run — the same rule
    `entity_merges` applies to a resolution nobody accepted.

    `subject_key` is what makes re-running discovery idempotent. A pattern is
    identified by *what it is about* — this assumption group, this confidence
    band — rather than by its sentence, because the sentence carries the current
    numbers and would change every time the corpus grew. Without it a weekly
    `patterns discover` would leave a row per week saying almost the same thing.
    """

    __tablename__ = "patterns"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    # Which rule produced it, and what it is about. Together they are the
    # identity a re-run updates rather than duplicates.
    detector: Mapped[str] = mapped_column(Text, nullable=False)
    subject_key: Mapped[str] = mapped_column(Text, nullable=False)
    # Distinct decisions agreeing. Denormalised from `pattern_evidence` because
    # every read of this table sorts and filters on it, and recomputing a count
    # per row to render a list is how a page gets slow — but written only by the
    # code that writes the evidence, in the same transaction.
    support_count: Mapped[int] = mapped_column(Integer, nullable=False)
    contradiction_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )
    confidence: Mapped[float | None] = mapped_column(REAL)
    # The span the supporting decisions cover. A pattern drawn from three
    # decisions made in one afternoon is a different claim from one drawn from
    # three across a year, and the dates are what let a reader tell.
    first_observed: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    last_observed: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    discovered_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    dismissed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    dismissed_reason: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        _enum_check("kind", PatternKind, "ck_patterns_kind"),
        CheckConstraint("length(btrim(statement)) > 0", name="ck_patterns_statement"),
        # The rule that makes the citation requirement structural rather than a
        # convention in one module. A pattern claiming no support is a
        # behavioural claim with nothing behind it, and no writer may create one.
        CheckConstraint("support_count > 0", name="ck_patterns_support_positive"),
        CheckConstraint(
            "contradiction_count >= 0", name="ck_patterns_contradictions_non_negative"
        ),
        # And the rule that keeps confirmation bias out of the table itself:
        # more evidence must agree than disagree, or it is not a pattern.
        CheckConstraint(
            "support_count > contradiction_count",
            name="ck_patterns_support_exceeds_contradiction",
        ),
        CheckConstraint(
            "confidence IS NULL OR confidence BETWEEN 0.0 AND 1.0",
            name="ck_patterns_confidence_range",
        ),
        # A dismissal has a reason. "Rejected, no reason recorded" is a row
        # nobody can act on later, and the next run would have no way to tell a
        # considered refusal from a stale one.
        CheckConstraint(
            "(dismissed_at IS NULL) = (dismissed_reason IS NULL)",
            name="ck_patterns_dismissal_pairing",
        ),
        CheckConstraint(
            "first_observed IS NULL OR last_observed IS NULL "
            "OR first_observed <= last_observed",
            name="ck_patterns_observation_order",
        ),
        UniqueConstraint(
            "detector", "subject_key", name="uq_patterns_detector_subject"
        ),
        Index("ix_patterns_kind", "kind"),
    )


class PatternEvidence(Base):
    """One decision that agrees with a pattern, or argues against it.

    **Counter-evidence is a first-class row here, not a footnote.** A detector
    that stored only agreeing cases would make every candidate look strong,
    because the query that found it only looked for cases that fit. So each
    detector runs a second search for the decisions that contradict its
    candidate, those land here with `relation = 'contradicts'`, they lower the
    confidence through `domain/patterns.pattern_confidence`, and the interface
    shows them at the same weight as the supporting kind.

    `assumption_id` and `outcome_id` are nullable because the four detectors
    cite different things: an assumption pattern points at the assumption that
    broke, a calibration pattern at the outcome that resolved, a choice pattern
    at neither and only at the decision. All three cascade, and all three are
    inside the derived-table problem the phase already knows about — see
    `application/replay.py`.
    """

    __tablename__ = "pattern_evidence"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    pattern_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey("patterns.id", name="fk_pattern_evidence_pattern_id", ondelete="CASCADE"),
        nullable=False,
    )
    decision_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id", name="fk_pattern_evidence_decision_id", ondelete="CASCADE"
        ),
        nullable=False,
    )
    assumption_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "decision_assumptions.id",
            name="fk_pattern_evidence_assumption_id",
            ondelete="CASCADE",
        ),
    )
    outcome_id: Mapped[UUID | None] = mapped_column(
        _UUID,
        ForeignKey(
            "decision_outcomes.id",
            name="fk_pattern_evidence_outcome_id",
            ondelete="CASCADE",
        ),
    )
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    # Why this decision counts for or against. Written by the detector and shown
    # verbatim, because "supports" beside a decision title is not something a
    # reader can check and "the assumption 'deployment takes two days' failed
    # here" is.
    note: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint(
            "pattern_id",
            "decision_id",
            "assumption_id",
            "outcome_id",
            name="uq_pattern_evidence_link",
            postgresql_nulls_not_distinct=True,
        ),
        _enum_check("relation", PatternRelation, "ck_pattern_evidence_relation"),
        Index("ix_pattern_evidence_pattern", "pattern_id", "relation"),
        Index("ix_pattern_evidence_decision", "decision_id"),
    )


class Reflection(Base):
    """A pattern, in prose, with the citations that make it checkable.

    **The riskiest row in this schema, and it is worth saying why it is riskier
    than `patterns` — which the docstring above already called the most
    dangerous table.** A pattern is a statement assembled from counts, sitting
    beside the decisions it was counted from, and a reader looks at both
    together. This is fluent English about somebody's judgement, and prose is
    read as a claim rather than as a summary of a table. Everything else this
    system stores is retrieved text or arithmetic; this is the one row a
    language model wrote.

    So three columns exist to keep it answerable rather than merely readable.

    `citation_rate` is what `domain/grounding.check_reflection` measured at
    generation: cited sentences over sentences, where *every* sentence must
    cite, not only the ones a heuristic calls factual. Stored rather than
    recomputed on read because it is a fact about the text as generated, and a
    number recomputed later against a pattern whose evidence has since moved
    would be a different measurement wearing the same name.

    `model_id` is which model wrote it. A reflection is the only row here whose
    wording is not reproducible from the data, so the thing that produced the
    wording is part of the record.

    `dismissed_at` is the column that matters most. You have to be able to say
    "this is wrong about me" and have the system stop repeating it — and stop
    *regenerating* it, which is why `application/reflections.py` refuses to
    generate for a pattern whose reflection was dismissed rather than only
    hiding the row. A rejection a weekly re-run undid would not be a rejection.

    `acknowledged_at` is deliberately not the same thing and deliberately not a
    verdict. It records that somebody read it, so a view can stop showing an
    unread claim first; it is not agreement, and nothing downstream weights a
    reflection by it.
    """

    __tablename__ = "reflections"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    pattern_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey("patterns.id", name="fk_reflections_pattern_id", ondelete="CASCADE"),
        nullable=False,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    citation_rate: Mapped[float | None] = mapped_column(REAL)
    generated_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    model_id: Mapped[str] = mapped_column(Text, nullable=False)
    acknowledged_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    dismissed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    dismissed_reason: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("length(btrim(text)) > 0", name="ck_reflections_text"),
        CheckConstraint(
            "citation_rate IS NULL OR citation_rate BETWEEN 0.0 AND 1.0",
            name="ck_reflections_citation_rate_range",
        ),
        # The same rule `patterns` carries: a rejection without a reason is a row
        # nobody can tell from a stale one.
        CheckConstraint(
            "(dismissed_at IS NULL) = (dismissed_reason IS NULL)",
            name="ck_reflections_dismissal_pairing",
        ),
        Index("ix_reflections_pattern", "pattern_id"),
    )


class ReflectionCitation(Base):
    """One `[n]` in a reflection, resolved to the decision it points at.

    **A table rather than a column, and the reason is M1.4a.** That milestone
    stored citations as offsets, the offsets drifted under the text they pointed
    into, nothing failed — row counts were right, every test passed — and
    highlights pointed a few hundred characters from the answer. The only thing
    that catches that class of bug is making the identity structural.

    The same drift is available here and it is worse. A reflection's `[3]` means
    "the third decision in the list this reflection was generated from", and
    `patterns discover` **replaces a pattern's evidence wholesale** on every
    re-run — so re-deriving the numbering at read time would silently renumber
    every citation the first time the corpus grew. The claim would still read
    correctly and would link to a different decision.

    So the numbering is frozen here at generation, as a foreign key. A cited
    decision that is later deleted takes its citation with it rather than
    leaving a marker pointing at nothing.

    `relation` is carried across from `pattern_evidence` rather than joined back
    to it, for the same reason: it is what the *reflection* was told about this
    decision, and the interface uses it to show that the counter-evidence really
    was cited in the same paragraph as the claim. Re-reading it from the pattern
    later would report what the detector thinks today.
    """

    __tablename__ = "reflection_citations"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    reflection_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "reflections.id",
            name="fk_reflection_citations_reflection_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    # The number as it appears in the text. 1-based, matching what the model was
    # shown.
    marker: Mapped[int] = mapped_column(Integer, nullable=False)
    decision_id: Mapped[UUID] = mapped_column(
        _UUID,
        ForeignKey(
            "decisions.id",
            name="fk_reflection_citations_decision_id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    relation: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint("reflection_id", "marker", name="uq_reflection_citations_marker"),
        CheckConstraint("marker > 0", name="ck_reflection_citations_marker_positive"),
        _enum_check("relation", PatternRelation, "ck_reflection_citations_relation"),
        Index("ix_reflection_citations_decision", "decision_id"),
    )


class ExternalEvent(Base):
    """Something that happened outside, and the trigger for work nobody asked for.

    **The first table in this schema whose rows arrive rather than being
    written.** Everything before it is the product of somebody using the system:
    a source they registered, a decision they recorded, a query they judged. This
    is a plugin firing on a keystroke, and the difference shows up in the two
    indexes below rather than in the columns.

    Named `ExternalEvent` in Python and `events` in SQL. `IngestionEvent`
    already holds the other meaning of the word — M1.1's append-only log of what
    was observed at a source, which is source-of-truth and never discarded — and
    two classes called `Event` in one schema is how somebody truncates the wrong
    one.

    **Bitemporal for M1.1's reason, unchanged.** `occurred_at` is when the thing
    happened and `received_at` is when this system heard about it; an editor
    event delivered after a network hiccup happened earlier than it arrived, and
    collapsing the two would silently re-date every event to whenever the
    connection recovered. There is no `occurred_at_source` here because there is
    only one way an event gets its time — the client asserts it — and a client
    that asserts nothing gets `received_at`, which makes the two equal. That
    equality *is* the provenance: it means nobody said when.

    `processed_at` is null until a handler has run to completion, and it is
    deliberately not set at dispatch. An event whose jobs are all still pending
    has had nothing done about it, and marking it processed when the jobs were
    enqueued would make the latency in `events stats` measure the speed of an
    INSERT.

    `payload` is JSONB and is not validated beyond being an object. M6.0 is
    plumbing: no handler reads a field of it yet, and a schema invented before
    the first consumer exists is a schema that will be wrong when one does.
    """

    __tablename__ = "events"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    # Which client emitted it. Free text rather than an enum: clients are added
    # by whoever writes a plugin, and an enum here would mean a migration before
    # a new editor could say anything at all.
    source: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    occurred_at: Mapped[datetime] = mapped_column(_TIMESTAMPTZ, nullable=False)
    received_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    processed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    dedupe_key: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        _enum_check("kind", EventKind, "ck_events_kind"),
        CheckConstraint("length(btrim(source)) > 0", name="ck_events_source"),
        # The rule that makes ten keystrokes one unit of work, and the `WHERE`
        # is the whole of it. Restricted to unprocessed rows, so the same file
        # focused again tomorrow is new work rather than a permanent collision —
        # an index over every row would refuse the second focus forever.
        Index(
            "uq_events_pending_dedupe",
            "kind",
            "dedupe_key",
            unique=True,
            postgresql_where=text("processed_at IS NULL AND dedupe_key IS NOT NULL"),
        ),
        # The rate limiter's query, and the reason it can afford to run on every
        # POST: counting one source's last minute is an index range scan.
        Index("ix_events_source_received", "source", "received_at"),
        # `events tail` and `events stats`, both of which read by kind in time
        # order.
        Index("ix_events_kind_received", "kind", "received_at"),
        # Finding the backlog without scanning processed history. Partial,
        # because the pending set is the only one anybody queries for.
        Index(
            "ix_events_unprocessed",
            "received_at",
            postgresql_where=text("processed_at IS NULL"),
        ),
    )


class ContextCache(Base):
    """An assembled context, kept so the next trigger does not rebuild it.

    **The only cache in this schema whose entries can be *wrong* rather than
    merely stale**, and the difference shapes every column. `embedding_cache` is
    content-addressed: an entry is a pure function of (model, role, text), so a
    retained one is correct by construction. A context is a function of the whole
    corpus, and a corpus that has changed makes every context built before it a
    confident answer to a question whose evidence has moved.

    So `cache_key` carries a fingerprint of the corpus, not just the focus. A
    sync that ingests one file changes the fingerprint, every key changes with
    it, and nothing has to know which focuses were affected — see
    `application/context_engine.corpus_fingerprint`. Over-invalidation costs a
    re-assembly measured in hundreds of milliseconds; under-invalidation serves
    context that omits the file somebody just changed, which is the failure the
    cache exists to avoid rather than to cause.

    `expires_at` is a second, weaker rule and answers a different question. The
    fingerprint handles staleness of *content*; this handles staleness of
    *intent*. A context assembled for a meeting three days ago is answering
    something nobody is asking now, and keeping it is a row that will never be
    read.

    `hit_count` is the number this milestone is judged on. Precomputing context
    for triggers that never get read burns compute continuously and produces
    mostly waste, so the hit rate is the evidence for whether precomputation
    earns its cost. Counted in the database on the read rather than in a log,
    because a rate assembled from log lines is a rate nobody can query later.

    `payload` is the rendered context rather than a list of ids. Storing ids
    would mean re-reading and re-rendering every item on a hit, which is most of
    the cost the cache exists to avoid — and it would let a hit return text that
    no longer matches what selection actually saw.
    """

    __tablename__ = "context_cache"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    # Focus plus budget plus the corpus fingerprint, hashed. Unique because it
    # *is* the identity: two rows under one key would mean two answers to the
    # same question with nothing to choose between them.
    cache_key: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    # Kept beside the key it is hashed into, so a person reading this table can
    # see what a row is for. A hash nobody can reverse is a row nobody can debug.
    focus: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=_EMPTY_JSONB
    )
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    built_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(_TIMESTAMPTZ, nullable=False)
    hit_count: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    __table_args__ = (
        CheckConstraint("length(btrim(focus)) > 0", name="ck_context_cache_focus"),
        CheckConstraint("token_count >= 0", name="ck_context_cache_tokens"),
        CheckConstraint("hit_count >= 0", name="ck_context_cache_hits"),
        # A row that expires before it was built is a clock or a TTL bug, and it
        # would be invisible otherwise: the entry simply never serves a hit.
        CheckConstraint("expires_at > built_at", name="ck_context_cache_expiry_order"),
        # The read path is `WHERE cache_key = ? AND expires_at > now()`, and the
        # unique index on `cache_key` already serves it. This one is for the
        # sweep that removes dead rows, which is otherwise a full scan of the
        # table most likely to accumulate them.
        Index("ix_context_cache_expires", "expires_at"),
    )


class SurfacingLog(Base):
    """Every decision to volunteer context, and every decision not to.

    **One row per decision rather than one per interruption**, which is the only
    thing here that is not what M6.3 asked for, and the reason is the question a
    push system has to be able to answer: *why didn't it show me anything?* A
    table of what was shown cannot answer it — silence looks the same whether the
    gate refused, the corpus was empty, or nothing ever ran. So `surfaced_at` is
    nullable, `reason` is not, and a refusal is a row carrying the score it
    reached and the bar it did not.

    `score` and `threshold` are both stored even though the threshold is a pure
    function of this focus's feedback. That function's *inputs* change: recompute
    it a week later and you get today's bar, not the one the decision was made
    under, and the log would quietly start disagreeing with itself.

    `item_keys` sits beside `context_hash` because the two answer different
    questions. The hash is identity — is this exactly what you were shown — and
    the keys are what makes similarity computable, since one item different is a
    different hash and the same interruption. `domain/surfacing.overlap` is the
    comparison.

    **Classified user-authored**, which is not where its origin would put it.
    Nothing rebuilds from this table and no replay reproduces a gate decision, so
    by that test it belongs with `events` in `OPERATIONAL_TABLES`. Two columns
    decide otherwise: `dismissed_at` and `acted_on_at` are a person's judgement,
    they exist nowhere else, and `application/surfacing.py` reads them to raise
    the bar on a focus whose context keeps being refused. Truncating this would
    un-dismiss every dismissal and reset every adapted threshold, so the next
    trigger would surface exactly what somebody had told it not to. Same shape as
    the argument `patterns` settled in M5.3.

    No foreign key to `events` despite `trigger_id`. `events` is operational and
    a replay truncates it, so the constraint would either cascade away somebody's
    dismissal or block the truncation. A dangling breadcrumb is the honest price
    of pointing at a discardable table.
    """

    __tablename__ = "surfacing_log"

    id: Mapped[UUID] = mapped_column(_UUID, primary_key=True)
    focus: Mapped[str] = mapped_column(Text, nullable=False)
    context_hash: Mapped[str] = mapped_column(Text, nullable=False)
    item_keys: Mapped[list[Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    # The best item that was not the file already open, so the log can say what
    # was surfaced without re-assembling a context whose corpus has moved on.
    top_key: Mapped[str | None] = mapped_column(Text)
    top_title: Mapped[str | None] = mapped_column(Text)
    # Double precision, unlike the REAL this schema uses for confidences. Those
    # are opinions on a 0-1 scale; these are the two sides of a comparison
    # somebody may re-run, and a value that does not round-trip exactly would
    # make the row disagree with the decision it records.
    score: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    trigger_kind: Mapped[str | None] = mapped_column(Text)
    trigger_id: Mapped[UUID | None] = mapped_column(_UUID)
    decided_at: Mapped[datetime] = mapped_column(
        _TIMESTAMPTZ, nullable=False, server_default=func.now()
    )
    surfaced_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    dismissed_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)
    acted_on_at: Mapped[datetime | None] = mapped_column(_TIMESTAMPTZ)

    __table_args__ = (
        CheckConstraint("length(btrim(focus)) > 0", name="ck_surfacing_focus"),
        CheckConstraint(
            "reason IN ('"
            + "', '".join(reason.value for reason in SurfaceReason)
            + "')",
            name="ck_surfacing_reason",
        ),
        # The verdict and the outcome are stored separately because the report
        # needs both, and two columns that must agree eventually will not.
        CheckConstraint(
            "(reason = 'cleared') = (surfaced_at IS NOT NULL)",
            name="ck_surfacing_reason_matches_outcome",
        ),
        CheckConstraint(
            "surfaced_at IS NOT NULL OR (dismissed_at IS NULL AND acted_on_at IS NULL)",
            name="ck_surfacing_feedback_needs_surfacing",
        ),
        # "Useful" and "dismissed" are the same click made two ways. A row
        # holding both would be counted twice in the dismissal rate, which is
        # the one number this milestone is judged on.
        CheckConstraint(
            "NOT (dismissed_at IS NOT NULL AND acted_on_at IS NOT NULL)",
            name="ck_surfacing_one_verdict",
        ),
        # Partial: refusals are the majority of this table by design, and none
        # of them suppress anything.
        Index(
            "ix_surfacing_focus_surfaced",
            "focus",
            "surfaced_at",
            postgresql_where=text("surfaced_at IS NOT NULL"),
        ),
        Index("ix_surfacing_decided", "decided_at"),
        Index("ix_surfacing_focus", "focus"),
    )
