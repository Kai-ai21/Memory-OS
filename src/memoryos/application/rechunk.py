"""Find and re-enqueue memories whose chunks were produced by an older chunker.

This is the payoff for putting the chunker's parameters in its version string:
improving the chunker becomes a query over `chunker_version` rather than a
corpus-wide rebuild. It is also a small rehearsal for M1.7's replay, which is
the same shape over the event log.
"""

from dataclasses import dataclass
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.domain.jobs import JobSpec, JobType

logger = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class StaleMemory:
    id: UUID
    external_key: str


async def find_stale(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    current_version: str,
    source: str | None = None,
    stale_version: str | None = None,
) -> list[StaleMemory]:
    """Current memories holding chunks from a different chunker.

    `stale_version` narrows it to one exact version; without it, anything that
    is not the current version counts. Tombstoned memories are excluded — there
    is no point re-chunking something that has been deleted.
    """
    chunk_filter = (
        models.MemoryChunk.chunker_version != current_version
        if stale_version is None
        else models.MemoryChunk.chunker_version == stale_version
    )
    stale_chunks = select(models.MemoryChunk.memory_id).where(chunk_filter)

    stmt = (
        select(models.Memory.id, models.Memory.external_key)
        .where(
            models.Memory.is_current.is_(True),
            models.Memory.deleted_at.is_(None),
            models.Memory.id.in_(stale_chunks),
        )
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
        return [StaleMemory(id=row[0], external_key=row[1]) for row in rows]


async def enqueue_rechunk(
    session_factory: async_sessionmaker[AsyncSession], stale: list[StaleMemory]
) -> int:
    """Queue normalization for each stale memory, returning how many were queued.

    Fewer than `len(stale)` means the dedupe index collapsed some into work
    that was already pending — which is the correct outcome, not a failure.
    """
    enqueued = 0
    for memory in stale:
        async with session_factory.begin() as session:
            job_id = await enqueue_in(
                session,
                JobSpec(
                    job_type=JobType.NORMALIZE_MEMORY,
                    payload={"memory_id": str(memory.id)},
                    dedupe_key=str(memory.id),
                ),
            )
        enqueued += job_id is not None
    return enqueued
