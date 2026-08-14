"""One memory, with its chunks and its version history.

**This module exists because M7.0 went looking for the use case behind
`GET /memories/{id}` and there wasn't one.** The query — memory, source name,
chunks in ordinal order, every version newest first — lived in the route
handler, which is the one place a second caller cannot reach. That is precisely
the shape the milestone said to report: a capability written for a transport
rather than for the domain, invisible until something other than HTTP wanted it.

Nothing here is new behaviour. It is the route's own three queries, moved down a
layer and given a return type, and the route now calls it. The tool calls it too.
The test that mattered — `test_api_ui_endpoints` — did not change, because the
endpoint's output did not.

`get_current` on the repository is the neighbouring function and does something
different: it finds the *current* version by `(source, external_key)`, for the
ingestion path. This is by id, including superseded versions, for a reader.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models


class UnknownMemory(LookupError):
    """No memory with that id. A `LookupError` so the route can map it to 404."""


@dataclass(frozen=True, slots=True)
class ChunkRow:
    id: UUID
    ordinal: int
    content: str
    token_count: int
    char_start: int
    char_end: int
    prefix_chars: int
    chunker_version: str
    content_hash: str
    embedding_model: str | None
    embedded_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)
    embedded: bool = False


@dataclass(frozen=True, slots=True)
class VersionRow:
    id: UUID
    version: int
    is_current: bool
    content_hash: str
    normalized_hash: str | None
    occurred_at: datetime | None
    occurred_at_source: str
    ingested_at: datetime
    deleted_at: datetime | None


@dataclass(frozen=True, slots=True)
class MemoryDetail:
    """A memory as a reader wants it: its text, its spans, and its history."""

    id: UUID
    source_id: UUID
    source_name: str
    external_key: str
    version: int
    is_current: bool
    kind: str
    title: str | None
    content: str | None
    content_hash: str
    normalized_hash: str | None
    occurred_at: datetime | None
    occurred_at_source: str
    ingested_at: datetime
    deleted_at: datetime | None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: list[ChunkRow] = field(default_factory=list)
    # Every version of this item, newest first, so a caller can show what the
    # corpus knows about its history rather than only its present.
    versions: list[VersionRow] = field(default_factory=list)


async def show(
    sessions: async_sessionmaker[AsyncSession], memory_id: UUID
) -> MemoryDetail:
    """One memory by id, with chunks in ordinal order and every version.

    Ordinal order is load-bearing rather than tidy: a reader widens a hit into
    its surroundings by reading the neighbouring ordinals, so an arbitrary order
    here would quietly show the wrong surrounding text.

    Versions are keyed on `(source_id, external_key)` rather than on any
    self-reference, because that pair is what identifies "the same item" across
    re-ingestion — a memory id names one version and changes every time the file
    does.
    """
    async with sessions() as session:
        row = (
            await session.execute(
                select(models.Memory, models.Source.name)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Memory.id == memory_id)
            )
        ).one_or_none()
        if row is None:
            raise UnknownMemory(f"no memory {memory_id}")
        memory, source_name = row

        chunks = list(
            (
                await session.execute(
                    select(models.MemoryChunk)
                    .where(models.MemoryChunk.memory_id == memory_id)
                    .order_by(models.MemoryChunk.ordinal)
                )
            ).scalars()
        )
        versions = list(
            (
                await session.execute(
                    select(models.Memory)
                    .where(
                        models.Memory.source_id == memory.source_id,
                        models.Memory.external_key == memory.external_key,
                    )
                    .order_by(models.Memory.version.desc())
                )
            ).scalars()
        )

    return MemoryDetail(
        id=memory.id,
        source_id=memory.source_id,
        source_name=source_name,
        external_key=memory.external_key,
        version=memory.version,
        is_current=memory.is_current,
        kind=memory.kind,
        title=memory.title,
        content=memory.content,
        content_hash=memory.content_hash,
        normalized_hash=memory.normalized_hash,
        occurred_at=memory.occurred_at,
        occurred_at_source=memory.occurred_at_source,
        ingested_at=memory.ingested_at,
        deleted_at=memory.deleted_at,
        metadata=dict(memory.meta),
        chunks=[
            ChunkRow(
                id=chunk.id,
                ordinal=chunk.ordinal,
                content=chunk.content,
                token_count=chunk.token_count,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                prefix_chars=chunk.prefix_chars,
                chunker_version=chunk.chunker_version,
                content_hash=chunk.content_hash,
                embedding_model=chunk.embedding_model,
                embedded_at=chunk.embedded_at,
                metadata=dict(chunk.meta),
                embedded=chunk.embedding is not None,
            )
            for chunk in chunks
        ],
        versions=[
            VersionRow(
                id=version.id,
                version=version.version,
                is_current=version.is_current,
                content_hash=version.content_hash,
                normalized_hash=version.normalized_hash,
                occurred_at=version.occurred_at,
                occurred_at_source=version.occurred_at_source,
                ingested_at=version.ingested_at,
                deleted_at=version.deleted_at,
            )
            for version in versions
        ],
    )
