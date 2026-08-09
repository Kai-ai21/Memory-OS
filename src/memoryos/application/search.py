"""Semantic search: chunks are what match, memories are what you get back."""

import asyncio
import time
from dataclasses import dataclass, field
from statistics import mean
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import Embedder, ScoredChunk, SearchFilters, VectorStore
from memoryos.domain.values import MemoryKind

logger = structlog.get_logger(__name__)

# Chunks fetched per requested memory. Several chunks of one document commonly
# match together, so retrieving only k would routinely yield fewer than k
# distinct memories.
CHUNK_FANOUT = 5


@dataclass(frozen=True, slots=True)
class MemoryHit:
    memory_id: UUID
    external_key: str
    title: str | None
    kind: MemoryKind
    occurred_at: object
    score: float
    matched_chunks: list[ScoredChunk]


@dataclass(slots=True)
class SearchTiming:
    embed_ms: int = 0
    search_ms: int = 0
    total_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "embed_ms": self.embed_ms,
            "search_ms": self.search_ms,
            "total_ms": self.total_ms,
        }


@dataclass(slots=True)
class SearchResult:
    query: str
    hits: list[MemoryHit] = field(default_factory=list)
    timing: SearchTiming = field(default_factory=SearchTiming)


class SearchMemories:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        vector_store: VectorStore,
    ) -> None:
        self._sessions = session_factory
        self._embedder = embedder
        self._store = vector_store

    async def __call__(
        self,
        query: str,
        *,
        k: int = 10,
        filters: SearchFilters | None = None,
        ef_search: int | None = None,
        exact: bool = False,
    ) -> SearchResult:
        started = time.monotonic()
        resolved = filters or SearchFilters()

        embed_started = time.monotonic()
        # Same reasoning as the embed pipeline: CPU-bound matrix work does not
        # belong on the event loop, even for a single short query.
        #
        # `embed_query`, not `embed_passage`: this text is what the caller is
        # searching *with*, and the model was trained to see the two differently.
        (vector,) = await asyncio.to_thread(self._embedder.embed_query, [query])
        embed_ms = _elapsed_ms(embed_started)

        search_started = time.monotonic()
        wanted = max(k * CHUNK_FANOUT, k)
        if exact:
            chunks = await self._store.search_exact(vector, k=wanted, filters=resolved)
        else:
            chunks = await self._store.search(
                vector, k=wanted, filters=resolved, ef_search=ef_search
            )
        search_ms = _elapsed_ms(search_started)

        hits = await self._to_hits(chunks, k=k)

        timing = SearchTiming(
            embed_ms=embed_ms, search_ms=search_ms, total_ms=_elapsed_ms(started)
        )
        logger.info(
            "search.finished",
            query_length=len(query),
            chunks=len(chunks),
            hits=len(hits),
            exact=exact,
            **timing.as_dict(),
        )
        return SearchResult(query=query, hits=hits, timing=timing)

    async def _to_hits(self, chunks: list[ScoredChunk], *, k: int) -> list[MemoryHit]:
        if not chunks:
            return []

        grouped: dict[UUID, list[ScoredChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.memory_id, []).append(chunk)

        metadata = await self._memory_metadata(list(grouped))

        hits: list[MemoryHit] = []
        for memory_id, matched in grouped.items():
            row = metadata.get(memory_id)
            if row is None:
                # Deleted between the index read and this lookup.
                continue
            external_key, title, kind, occurred_at = row
            hits.append(
                MemoryHit(
                    memory_id=memory_id,
                    external_key=external_key,
                    title=title,
                    kind=MemoryKind(kind),
                    occurred_at=occurred_at,
                    # Max, not mean. A long document with one perfectly
                    # relevant paragraph should outrank a short one that is
                    # vaguely on-topic throughout; mean would penalise the long
                    # document for the parts that are not about the query.
                    score=max(chunk.score for chunk in matched),
                    # Ordinal order, because the reader wants them in document
                    # order rather than score order once they open the item.
                    matched_chunks=sorted(matched, key=lambda chunk: chunk.ordinal),
                )
            )

        hits.sort(key=_ranking, reverse=True)
        return hits[:k]

    async def _memory_metadata(
        self, memory_ids: list[UUID]
    ) -> dict[UUID, tuple[str, str | None, str, object]]:
        stmt = select(
            models.Memory.id,
            models.Memory.external_key,
            models.Memory.title,
            models.Memory.kind,
            models.Memory.occurred_at,
        ).where(models.Memory.id.in_(memory_ids))

        async with self._sessions() as session:
            rows = await session.execute(stmt)
            return {row[0]: (row[1], row[2], row[3], row[4]) for row in rows}


def _ranking(hit: MemoryHit) -> tuple[float, float]:
    """Best chunk first, then mean as the tie-break.

    Two documents whose best chunk scores identically are separated by how much
    of the rest is also relevant — which is the one place a mean is the right
    summary.
    """
    return (hit.score, mean(chunk.score for chunk in hit.matched_chunks))


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
