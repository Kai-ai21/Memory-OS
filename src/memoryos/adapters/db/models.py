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

from memoryos.domain.jobs import DEFAULT_MAX_ATTEMPTS, JobStatus
from memoryos.domain.values import (
    HEX64_PATTERN,
    EntityType,
    EventType,
    MemoryKind,
    MergeStatus,
    MergeStrategy,
    Predicate,
    SourceKind,
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
