"""One memory, in full, with everything needed to read it.

`GET /memories` in `sources.py` lists identities. This is the other half: the
normalized text, every chunk in ordinal order, and the version history — enough
to draw chunk boundaries against real content, which is the diagnostic M2.0a
exists to make possible.
"""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.container import Container

router = APIRouter(tags=["memories"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class ChunkOut(BaseModel):
    """One chunk, with the provenance that makes it auditable.

    `char_start`/`char_end` index into the parent memory's `content`, which is
    what lets the UI draw boundaries on the document rather than concatenating
    chunk text and hoping it lines up.
    """

    id: UUID
    ordinal: int
    content: str
    token_count: int
    char_start: int
    char_end: int
    chunker_version: str
    content_hash: str
    embedding_model: str | None
    embedded_at: datetime | None
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Whether a vector exists, without shipping 384 floats to a browser that
    # would only count them.
    embedded: bool


class VersionOut(BaseModel):
    id: UUID
    version: int
    is_current: bool
    content_hash: str
    normalized_hash: str | None
    ingested_at: datetime
    deleted_at: datetime | None


class MemoryDetailOut(BaseModel):
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
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunks: list[ChunkOut]
    # Every version of this item, newest first, so the UI can show what the
    # corpus knows about its history rather than only its present.
    versions: list[VersionOut]


@router.get("/memories/{memory_id}", response_model=MemoryDetailOut)
async def get_memory(memory_id: UUID, container: ContainerDep) -> MemoryDetailOut:
    async with container.database.session_factory() as session:
        row = (
            await session.execute(
                select(models.Memory, models.Source.name)
                .join(models.Source, models.Source.id == models.Memory.source_id)
                .where(models.Memory.id == memory_id)
            )
        ).one_or_none()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such memory")
        memory, source_name = row

        chunks = list(
            (
                await session.execute(
                    select(models.MemoryChunk)
                    .where(models.MemoryChunk.memory_id == memory_id)
                    # Ordinal order, always. The UI reads neighbours by ordinal to
                    # widen a hit into its context, so an arbitrary order here
                    # would silently show the wrong surrounding text.
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

    return MemoryDetailOut(
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
            ChunkOut(
                id=chunk.id,
                ordinal=chunk.ordinal,
                content=chunk.content,
                token_count=chunk.token_count,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
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
            VersionOut(
                id=version.id,
                version=version.version,
                is_current=version.is_current,
                content_hash=version.content_hash,
                normalized_hash=version.normalized_hash,
                ingested_at=version.ingested_at,
                deleted_at=version.deleted_at,
            )
            for version in versions
        ],
    )
