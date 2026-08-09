"""pgvector-backed approximate and exact nearest-neighbour search."""

from collections.abc import Sequence
from typing import Any

import structlog
from sqlalchemy import ColumnElement, Select, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import Embedder, ScoredChunk, SearchFilters, VectorStore

logger = structlog.get_logger(__name__)

# The index is asked for more than the caller wants, because filtering happens
# after the index has already chosen its candidates. Postgres may scan-then-
# filter or filter-then-scan depending on how selective it believes the
# predicates are; in the first case a restrictive filter can leave far fewer
# than `k` rows standing.
#
# 2x rather than 4x: the search use case already asks for k*5 chunks to get k
# memories, so the two multipliers compound. At 4x a k=100 request scanned
# 2,000 candidates; at 2x it scans 1,000. The warning below is what catches
# the case where 2x was not enough.
OVER_FETCH = 2


class NotNormalized(RuntimeError):
    """The embedder does not produce unit vectors, so `vector_ip_ops` is wrong."""


class PgVectorStore(VectorStore):
    """Search over `memory_chunks.embedding`.

    Scores are similarities where higher is better. pgvector's `<#>` returns
    *negative* inner product, so it is negated here — a sign error would return
    the least similar chunks and look completely plausible in the response
    shape, which is why a test asserts that a chunk matched against its own
    text scores near 1.0.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        *,
        default_ef_search: int = 100,
    ) -> None:
        if not embedder.normalizes:
            # Loud, at construction. With non-normalized vectors inner product
            # ranks by magnitude as much as by direction, and pgvector will not
            # complain — it will just return a wrong ordering forever.
            raise NotNormalized(
                f"{embedder.model_id} does not produce unit vectors, but the HNSW "
                "index uses vector_ip_ops. Rebuild the index with vector_cosine_ops "
                "or use an embedder that normalizes."
            )
        self._sessions = session_factory
        self._default_ef_search = default_ef_search

    async def search(
        self,
        vector: Sequence[float],
        *,
        k: int,
        filters: SearchFilters,
        ef_search: int | None = None,
    ) -> list[ScoredChunk]:
        effective = ef_search or self._default_ef_search
        stmt = self._query(vector, limit=k * OVER_FETCH, filters=filters)

        async with self._sessions() as session:
            # LOCAL, so it dies with the transaction. A session-level SET would
            # ride the pooled connection into every later query on it, silently
            # changing the recall/latency trade for unrelated callers.
            # SET is a utility statement and takes no bind parameters, so the
            # value is coerced to int and interpolated — which is also what
            # makes that safe.
            await session.execute(text(f"SET LOCAL hnsw.ef_search = {int(effective)}"))
            rows = (await session.execute(stmt)).all()
            await session.rollback()

        found = _to_chunks(rows)
        if len(found) < k:
            # Visible rather than silent: the caller asked for k and the filter
            # ate the candidates the index had already committed to.
            logger.warning(
                "vector_store.under_fetched",
                requested=k,
                returned=len(found),
                over_fetch=OVER_FETCH,
                ef_search=effective,
            )
        return found[:k]

    async def search_exact(
        self, vector: Sequence[float], *, k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        """Sequential scan. Ground truth, not a production path."""
        stmt = self._query(vector, limit=k, filters=filters)

        async with self._sessions() as session:
            # Forces the planner past the HNSW index, so this is exhaustive by
            # construction rather than by hoping the planner agrees.
            await session.execute(text("SET LOCAL enable_indexscan = off"))
            await session.execute(text("SET LOCAL enable_indexonlyscan = off"))
            rows = (await session.execute(stmt)).all()
            await session.rollback()

        return _to_chunks(rows)

    def _query(
        self, vector: Sequence[float], *, limit: int, filters: SearchFilters
    ) -> Select[Any]:
        # pgvector's own comparator, so `<#>` is typed as returning a float
        # rather than inheriting the column's vector type — and the query
        # vector is a bound parameter rather than interpolated text.
        distance = models.MemoryChunk.embedding.max_inner_product(list(vector))

        stmt = (
            select(
                models.MemoryChunk.id,
                models.MemoryChunk.memory_id,
                models.MemoryChunk.ordinal,
                models.MemoryChunk.content,
                models.MemoryChunk.char_start,
                models.MemoryChunk.char_end,
                models.MemoryChunk.meta,
                # Negated: `<#>` is negative inner product, and callers want a
                # similarity where higher is better.
                (-distance).label("score"),
            )
            .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
            .where(*_predicates(filters))
            .order_by(distance)
            .limit(limit)
        )
        return stmt


def _predicates(filters: SearchFilters) -> list[ColumnElement[bool]]:
    """Everything a result must satisfy besides being close.

    `is_current` is unconditional: a superseded version's chunks describe text
    that is no longer what the item says, and surfacing them would be a
    correctness bug rather than a ranking one.
    """
    clauses: list[ColumnElement[bool]] = [
        # A chunk with no vector cannot be compared. Excluded rather than
        # allowed to blow up mid-scan.
        models.MemoryChunk.embedding.is_not(None),
        models.Memory.is_current.is_(True),
    ]

    if not filters.include_deleted:
        clauses.append(models.Memory.deleted_at.is_(None))
    if filters.source_ids:
        clauses.append(models.Memory.source_id.in_(list(filters.source_ids)))
    if filters.kinds:
        clauses.append(models.Memory.kind.in_([kind.value for kind in filters.kinds]))
    if filters.occurred_after is not None:
        clauses.append(models.Memory.occurred_at >= filters.occurred_after)
    if filters.occurred_before is not None:
        clauses.append(models.Memory.occurred_at <= filters.occurred_before)

    return clauses


def _to_chunks(rows: Sequence[Any]) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk_id=row[0],
            memory_id=row[1],
            ordinal=row[2],
            text=row[3],
            char_start=row[4],
            char_end=row[5],
            metadata=dict(row[6] or {}),
            score=float(row[7]),
        )
        for row in rows
    ]
