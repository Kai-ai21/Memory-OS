"""Fill chunk embeddings from the model, batched and cached."""

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum, auto
from uuid import UUID

import structlog
from sqlalchemy import func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import cache_key_for
from memoryos.application.ports import CacheEntry, Embedder, EmbeddingCache
from memoryos.domain.jobs import PermanentError
from memoryos.domain.values import EmbeddingRole

logger = structlog.get_logger(__name__)

# Everything this use case writes is stored text being indexed, never a query.
_ROLE = EmbeddingRole.PASSAGE


class EmbedOutcome(StrEnum):
    SKIPPED = auto()
    EMBEDDED = auto()
    DELETED = auto()


@dataclass(slots=True)
class EmbedReport:
    outcome: EmbedOutcome
    chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    duration_ms: int = 0

    @property
    def hit_rate(self) -> float:
        looked_up = self.cache_hits + self.cache_misses
        return self.cache_hits / looked_up if looked_up else 0.0

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "outcome": self.outcome.value,
            "chunks": self.chunks,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "duration_ms": self.duration_ms,
        }


class EmbedMemory:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        cache: EmbeddingCache,
        batch_size: int = 32,
    ) -> None:
        self._sessions = session_factory
        self._embedder = embedder
        self._cache = cache
        self._batch_size = max(1, batch_size)

    @property
    def model_id(self) -> str:
        return self._embedder.model_id

    async def __call__(self, memory_id: UUID) -> EmbedReport:
        started = time.monotonic()
        log = logger.bind(memory_id=str(memory_id), model=self._embedder.model_id)

        memory = await self._load(memory_id)
        if memory is None:
            raise PermanentError(f"no such memory: {memory_id}")
        if memory.deleted_at is not None:
            log.info("embed.skipped_deleted")
            return _finish(EmbedReport(EmbedOutcome.DELETED), started)

        pending = await self._pending_chunks(memory_id)
        if not pending:
            # Every chunk already carries a vector from this exact model.
            log.info("embed.skipped_current")
            return _finish(EmbedReport(EmbedOutcome.SKIPPED), started)

        report = await self._embed_chunks(pending)
        log.info("embed.finished", **report.as_dict())
        return _finish(report, started)

    async def _load(self, memory_id: UUID) -> models.Memory | None:
        async with self._sessions() as session:
            return await session.get(models.Memory, memory_id)

    async def _pending_chunks(self, memory_id: UUID) -> list[tuple[UUID, str]]:
        """Chunks with no vector, or a vector from a different model.

        Both halves matter. The first is the ordinary backfill; the second is
        what makes a model upgrade actually re-embed instead of looking at a
        populated column and deciding there is nothing to do.
        """
        stmt = (
            select(models.MemoryChunk.id, models.MemoryChunk.content)
            .where(
                models.MemoryChunk.memory_id == memory_id,
                or_(
                    models.MemoryChunk.embedding.is_(None),
                    models.MemoryChunk.embedding_model != self._embedder.model_id,
                ),
            )
            .order_by(models.MemoryChunk.ordinal)
        )
        async with self._sessions() as session:
            return [(row[0], row[1]) for row in await session.execute(stmt)]

    async def _embed_chunks(self, pending: list[tuple[UUID, str]]) -> EmbedReport:
        model_id = self._embedder.model_id
        # Passages, always. These rows are what search compares *against*; a
        # query vector written here would be in the wrong half of an asymmetric
        # model's geometry and nothing downstream could tell.
        keys = [cache_key_for(model_id, text, role=_ROLE) for _, text in pending]

        cached = await self._cache.get_many(list(dict.fromkeys(keys)))

        # Deduplicated within this memory too: two identical chunks are one
        # forward pass, not two.
        missing_texts: list[str] = []
        seen: set[str] = set()
        for key, (_, text) in zip(keys, pending, strict=True):
            if key not in cached and key not in seen:
                seen.add(key)
                missing_texts.append(text)

        fresh: dict[str, list[float]] = {}
        for batch in _batched(missing_texts, self._batch_size):
            # CPU-bound matrix multiplication. On the event loop it would stall
            # every other coroutine in the process, health checks included.
            vectors = await asyncio.to_thread(self._embedder.embed_passage, batch)
            self._check_widths(vectors)
            for text, vector in zip(batch, vectors, strict=True):
                fresh[cache_key_for(model_id, text, role=_ROLE)] = vector

        if fresh:
            await self._cache.put_many(
                [
                    CacheEntry(
                        cache_key=key,
                        model_id=model_id,
                        dimension=self._embedder.dimension,
                        embedding=vector,
                    )
                    for key, vector in fresh.items()
                ]
            )

        vectors_by_key = {**cached, **fresh}
        await self._write(pending, keys, vectors_by_key)

        return EmbedReport(
            outcome=EmbedOutcome.EMBEDDED,
            chunks=len(pending),
            cache_hits=sum(1 for key in keys if key in cached),
            cache_misses=len(missing_texts),
        )

    def _check_widths(self, vectors: list[list[float]]) -> None:
        """Refuse a wrong-width vector before it reaches the database.

        pgvector would reject it too, but at insert time and with an error that
        names neither the model nor the chunk — a long way from the cause.
        """
        expected = self._embedder.dimension
        for vector in vectors:
            if len(vector) != expected:
                raise PermanentError(
                    f"{self._embedder.model_id} returned a {len(vector)}-dimensional "
                    f"vector, expected {expected}"
                )

    async def _write(
        self,
        pending: list[tuple[UUID, str]],
        keys: list[str],
        vectors_by_key: dict[str, list[float]],
    ) -> None:
        """Attach the vectors to their chunks, in one transaction.

        The cache is written first and separately on purpose: if this
        transaction fails, the vectors survive, and the retried job finds them
        all in cache and does no model work at all.
        """
        async with self._sessions.begin() as session:
            for (chunk_id, _), key in zip(pending, keys, strict=True):
                vector = vectors_by_key.get(key)
                if vector is None:
                    raise PermanentError(f"no vector produced for chunk {chunk_id}")
                await session.execute(
                    update(models.MemoryChunk)
                    .where(models.MemoryChunk.id == chunk_id)
                    .values(
                        embedding=vector,
                        embedding_model=self._embedder.model_id,
                        embedded_at=func.now(),
                    )
                )


def _batched(items: list[str], size: int) -> list[list[str]]:
    return [items[start : start + size] for start in range(0, len(items), size)]


def _finish(report: EmbedReport, started: float) -> EmbedReport:
    report.duration_ms = int((time.monotonic() - started) * 1000)
    return report
