"""Postgres implementations of the application ports.

Each repository takes an `AsyncSession` and does not commit. The session is the
unit of work and the caller owns the transaction boundary, which is what lets a
use case append an event and update a projection atomically.

The classes inherit their Protocol explicitly. Structural typing would accept
them either way; the explicit base makes mypy fail here, at the definition,
rather than at whichever call site first passes one to a use case.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from memoryos.adapters.db import models
from memoryos.adapters.db.mappers import (
    to_artifact_row,
    to_event,
    to_event_row,
    to_memory,
    to_memory_row,
    to_source,
    to_source_row,
)
from memoryos.application.ports import (
    ArtifactRepository,
    EventLog,
    MemoryRepository,
    SourceRepository,
)
from memoryos.domain.entities import IngestionEvent, Memory, RawArtifact, Source
from memoryos.domain.values import ContentHash, SourceKind


class SqlAlchemySourceRepository(SourceRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, source_id: UUID) -> Source | None:
        row = await self._session.get(models.Source, source_id)
        return None if row is None else to_source(row)

    async def get_by_name(self, kind: SourceKind, name: str) -> Source | None:
        stmt = select(models.Source).where(
            models.Source.kind == kind.value, models.Source.name == name
        )
        row = (await self._session.execute(stmt)).scalar_one_or_none()
        return None if row is None else to_source(row)

    async def add(self, source: Source) -> None:
        self._session.add(to_source_row(source))
        await self._session.flush()

    async def update_cursor(self, source_id: UUID, cursor: dict[str, Any]) -> None:
        await self._session.execute(
            update(models.Source)
            .where(models.Source.id == source_id)
            .values(cursor=cursor)
            .execution_options(synchronize_session="fetch")
        )


class SqlAlchemyArtifactRepository(ArtifactRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def exists(self, content_hash: ContentHash) -> bool:
        stmt = select(models.RawArtifact.content_hash).where(
            models.RawArtifact.content_hash == content_hash.value
        )
        return (await self._session.execute(stmt)).scalar_one_or_none() is not None

    async def add(self, artifact: RawArtifact) -> None:
        self._session.add(to_artifact_row(artifact))
        await self._session.flush()


class SqlAlchemyEventLog(EventLog):
    """Append and replay. There is no method here that rewrites history."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append(self, event: IngestionEvent) -> None:
        self._session.add(to_event_row(event))
        await self._session.flush()

    async def replay(self, after_seq: int = 0, limit: int = 1000) -> Sequence[IngestionEvent]:
        stmt = (
            select(models.IngestionEvent)
            .where(models.IngestionEvent.seq > after_seq)
            .order_by(models.IngestionEvent.seq)
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).scalars().all()
        return [to_event(row) for row in rows]


class SqlAlchemyMemoryRepository(MemoryRepository):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_current(self, source_id: UUID, external_key: str) -> Memory | None:
        row = await self._current_row(source_id, external_key)
        return None if row is None else to_memory(row)

    async def add_version(self, memory: Memory) -> None:
        """Supersede the current version and insert the next one.

        Both statements run in the caller's transaction, so either both land or
        neither does. They are flushed in two steps on purpose: the partial
        unique index on `(source_id, external_key) WHERE is_current` is not
        deferrable, so the old row has to stop being current before the new row
        starts.
        """
        current = await self._current_row(memory.source_id, memory.external_key)

        if current is None:
            next_version = 1
        else:
            next_version = current.version + 1
            current.is_current = False
            await self._session.flush()

        self._session.add(
            to_memory_row(replace(memory, version=next_version, is_current=True))
        )
        await self._session.flush()

    async def tombstone(self, memory_id: UUID) -> None:
        """Mark a memory deleted without removing the row.

        `is_current` is left alone: the tombstone is the current state of the
        item, and readers filter on `deleted_at`. Dropping the flag instead
        would make a deleted item indistinguishable from one never ingested.
        """
        await self._session.execute(
            update(models.Memory)
            .where(models.Memory.id == memory_id)
            .values(deleted_at=func.now())
            .execution_options(synchronize_session="fetch")
        )

    async def _current_row(self, source_id: UUID, external_key: str) -> models.Memory | None:
        stmt = select(models.Memory).where(
            models.Memory.source_id == source_id,
            models.Memory.external_key == external_key,
            models.Memory.is_current.is_(True),
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()
