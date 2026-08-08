"""Finding work for the embedder, and reporting what it has done.

`reembed` is the rehearsal for the model swap Phase 2 will demand. Making that
operation a query rather than a rebuild is most of the reason the model id is
in the cache key and on every chunk row.
"""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.domain.jobs import JobSpec, JobType


@dataclass(frozen=True, slots=True)
class PendingMemory:
    id: UUID
    external_key: str
    chunks: int


@dataclass(frozen=True, slots=True)
class Stats:
    memories: int
    current_memories: int
    chunks: int
    embedded_chunks: int
    cache_entries: int
    models: dict[str, int]

    @property
    def coverage(self) -> float:
        return self.embedded_chunks / self.chunks if self.chunks else 0.0


async def find_unembedded(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    model_id: str,
    source: str | None = None,
    stale_only: bool = False,
) -> list[PendingMemory]:
    """Memories with chunks this model has not embedded.

    `stale_only` narrows to chunks embedded by a *different* model, which is
    the re-embed case; otherwise a null embedding counts too, which is the
    ordinary backfill.
    """
    if stale_only:
        chunk_filter = models.MemoryChunk.embedding_model.is_distinct_from(model_id)
        chunk_filter = chunk_filter & models.MemoryChunk.embedding.is_not(None)
    else:
        chunk_filter = or_(
            models.MemoryChunk.embedding.is_(None),
            models.MemoryChunk.embedding_model.is_distinct_from(model_id),
        )

    stmt = (
        select(
            models.Memory.id,
            models.Memory.external_key,
            func.count(models.MemoryChunk.id),
        )
        .join(models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id)
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            chunk_filter,
        )
        .group_by(models.Memory.id, models.Memory.external_key)
        .order_by(models.Memory.external_key)
    )
    if source is not None:
        stmt = stmt.where(
            models.Memory.source_id.in_(
                select(models.Source.id).where(models.Source.name == source)
            )
        )

    async with session_factory() as session:
        rows = await session.execute(stmt)
        return [PendingMemory(id=row[0], external_key=row[1], chunks=row[2]) for row in rows]


async def enqueue_embedding(
    session_factory: async_sessionmaker[AsyncSession], pending: list[PendingMemory]
) -> int:
    enqueued = 0
    for memory in pending:
        async with session_factory.begin() as session:
            job_id = await enqueue_in(
                session,
                JobSpec(
                    job_type=JobType.EMBED_MEMORY,
                    payload={"memory_id": str(memory.id)},
                    dedupe_key=f"embed:{memory.id}",
                ),
            )
        enqueued += job_id is not None
    return enqueued


async def gather_stats(session_factory: async_sessionmaker[AsyncSession]) -> Stats:
    async with session_factory() as session:
        memories = (
            await session.execute(select(func.count()).select_from(models.Memory))
        ).scalar_one()
        current = (
            await session.execute(
                select(func.count())
                .select_from(models.Memory)
                .where(models.Memory.is_current.is_(True), models.Memory.deleted_at.is_(None))
            )
        ).scalar_one()
        chunks = (
            await session.execute(select(func.count()).select_from(models.MemoryChunk))
        ).scalar_one()
        embedded = (
            await session.execute(
                select(func.count())
                .select_from(models.MemoryChunk)
                .where(models.MemoryChunk.embedding.is_not(None))
            )
        ).scalar_one()
        cache_entries = (
            await session.execute(
                select(func.count()).select_from(models.EmbeddingCacheEntry)
            )
        ).scalar_one()
        by_model: dict[str, int] = {
            str(model or ""): count
            for model, count in (
                await session.execute(
                    select(models.MemoryChunk.embedding_model, func.count())
                    .where(models.MemoryChunk.embedding.is_not(None))
                    .group_by(models.MemoryChunk.embedding_model)
                )
            ).all()
        }

    return Stats(
        memories=memories,
        current_memories=current,
        chunks=chunks,
        embedded_chunks=embedded,
        cache_entries=cache_entries,
        models=by_model,
    )
