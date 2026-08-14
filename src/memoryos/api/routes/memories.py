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

from memoryos.application import memories as memories_app
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
    """One version, with both clocks and both hashes.

    M4.1 draws these on a small timeline, which needs `occurred_at` as well as
    `ingested_at` — the two are a different story per version, and a history
    showing only when the system read something cannot say when the thing it
    read was written.

    The hashes are what "what changed" is answered from. `content_hash`
    differing with `normalized_hash` identical is a real and common case — a
    trailing newline, a line ending, a byte the normalizer discards — and it is
    the difference between a version that changed the corpus and one that only
    changed the file.
    """

    id: UUID
    version: int
    is_current: bool
    content_hash: str
    normalized_hash: str | None
    occurred_at: datetime | None
    occurred_at_source: str
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
    """One memory, its chunks and its versions.

    **The three queries this used to run now live in `application/memories.py`.**
    M7.0 needed the same thing for a tool and found there was no use case to
    call — the whole capability was in this function body, reachable only over
    HTTP. Nothing about the response changed; the query moved down a layer and
    this became what a route should be, which is a mapping from a domain type to
    a wire type.
    """
    try:
        detail = await memories_app.show(container.database.session_factory, memory_id)
    except memories_app.UnknownMemory as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such memory") from exc

    return MemoryDetailOut(
        id=detail.id,
        source_id=detail.source_id,
        source_name=detail.source_name,
        external_key=detail.external_key,
        version=detail.version,
        is_current=detail.is_current,
        kind=detail.kind,
        title=detail.title,
        content=detail.content,
        content_hash=detail.content_hash,
        normalized_hash=detail.normalized_hash,
        occurred_at=detail.occurred_at,
        occurred_at_source=detail.occurred_at_source,
        ingested_at=detail.ingested_at,
        deleted_at=detail.deleted_at,
        metadata=detail.metadata,
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
                metadata=chunk.metadata,
                embedded=chunk.embedded,
            )
            for chunk in detail.chunks
        ],
        versions=[
            VersionOut(
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
            for version in detail.versions
        ],
    )
