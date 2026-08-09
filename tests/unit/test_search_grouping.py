"""Grouping chunks into memories, and the filter predicates, without a database."""

from datetime import UTC, datetime
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from memoryos.adapters.db import models
from memoryos.adapters.db.vector_store import NotNormalized, PgVectorStore, _predicates
from memoryos.application.ports import ScoredChunk, SearchFilters
from memoryos.application.search import MemoryHit, _ranking
from memoryos.domain.ids import new_id
from memoryos.domain.values import MemoryKind
from tests.support.fakes import FakeEmbedder


def chunk(memory_id: UUID, ordinal: int, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=new_id(),
        memory_id=memory_id,
        ordinal=ordinal,
        text=f"chunk {ordinal}",
        score=score,
        char_start=ordinal * 10,
        char_end=ordinal * 10 + 9,
    )


def hit(chunks: list[ScoredChunk]) -> MemoryHit:
    return MemoryHit(
        memory_id=chunks[0].memory_id,
        external_key="notes.md",
        source_name="corpus",
        title=None,
        kind=MemoryKind.NOTE,
        occurred_at=datetime(2024, 1, 1, tzinfo=UTC),
        score=max(c.score for c in chunks),
        matched_chunks=sorted(chunks, key=lambda c: c.ordinal),
    )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_a_memorys_score_is_its_best_chunk() -> None:
    """Max, not mean.

    A long document with one perfectly relevant paragraph should outrank a
    short one that is vaguely on-topic throughout. A mean would penalise the
    long document for the parts that are not about the query.
    """
    memory_id = new_id()
    chunks = [chunk(memory_id, 0, 0.30), chunk(memory_id, 1, 0.91), chunk(memory_id, 2, 0.22)]

    assert hit(chunks).score == pytest.approx(0.91)


def test_matched_chunks_come_back_in_ordinal_order() -> None:
    # Document order, not score order: once the reader opens the item they want
    # to read it forwards.
    memory_id = new_id()
    chunks = [chunk(memory_id, 2, 0.9), chunk(memory_id, 0, 0.5), chunk(memory_id, 1, 0.7)]

    assert [c.ordinal for c in hit(chunks).matched_chunks] == [0, 1, 2]


def test_ties_on_the_best_chunk_are_broken_by_the_mean() -> None:
    # Same best chunk; the one with more of the rest also relevant wins.
    focused = hit([chunk(new_id(), 0, 0.80), chunk(new_id(), 1, 0.10)])
    broad = hit([chunk(new_id(), 0, 0.80), chunk(new_id(), 1, 0.70)])

    assert _ranking(broad) > _ranking(focused)
    assert sorted([focused, broad], key=_ranking, reverse=True)[0] is broad


def test_a_single_chunk_memory_ranks_on_that_chunk() -> None:
    best, average = _ranking(hit([chunk(new_id(), 0, 0.75)]))
    assert best == pytest.approx(0.75)
    assert average == pytest.approx(0.75)


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


def rendered(filters: SearchFilters) -> str:
    stmt = select(models.MemoryChunk.id).where(*_predicates(filters))
    return str(stmt.compile(dialect=postgresql.dialect()))  # type: ignore[no-untyped-call]


def test_the_default_filter_excludes_deleted_and_superseded() -> None:
    sql = rendered(SearchFilters())

    # Unconditional: a superseded version's chunks describe text the item no
    # longer says, and a tombstoned memory should not surface at all.
    assert "memories.is_current IS true" in sql
    assert "memories.deleted_at IS NULL" in sql
    assert "memory_chunks.embedding IS NOT NULL" in sql


def test_include_deleted_drops_only_the_tombstone_predicate() -> None:
    sql = rendered(SearchFilters(include_deleted=True))

    assert "memories.deleted_at IS NULL" not in sql
    # is_current stays: including deleted items is not the same as including
    # stale versions of live ones.
    assert "memories.is_current IS true" in sql


def test_source_and_kind_filters_add_in_predicates() -> None:
    sql = rendered(
        SearchFilters(source_ids=[new_id()], kinds=[MemoryKind.NOTE, MemoryKind.CODE])
    )

    assert "memories.source_id IN" in sql
    assert "memories.kind IN" in sql


def test_time_bounds_add_range_predicates() -> None:
    sql = rendered(
        SearchFilters(
            occurred_after=datetime(2024, 1, 1, tzinfo=UTC),
            occurred_before=datetime(2024, 12, 31, tzinfo=UTC),
        )
    )

    assert "memories.occurred_at >=" in sql
    assert "memories.occurred_at <=" in sql


def test_absent_filters_add_no_predicates() -> None:
    minimal = rendered(SearchFilters())
    assert "source_id IN" not in minimal
    assert "kind IN" not in minimal
    assert "occurred_at" not in minimal


# --------------------------------------------------------------------------
# The normalization guard
# --------------------------------------------------------------------------


def test_a_non_normalizing_embedder_is_refused_at_construction() -> None:
    """`vector_ip_ops` is only equivalent to cosine for unit vectors.

    With non-normalized vectors, inner product ranks partly by magnitude and
    pgvector will not complain — it will just return a wrong ordering forever.
    Failing at construction is the only place this is cheap to notice.
    """

    class Unnormalized(FakeEmbedder):
        @property
        def normalizes(self) -> bool:
            return False

    with pytest.raises(NotNormalized, match="unit vectors"):
        PgVectorStore(None, Unnormalized())  # type: ignore[arg-type]
