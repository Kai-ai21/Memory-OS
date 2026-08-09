"""Semantic search over memories."""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.application.ports import SearchFilters
from memoryos.application.search import SearchResult
from memoryos.container import Container
from memoryos.domain.values import MemoryKind

router = APIRouter(tags=["search"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class ChunkOut(BaseModel):
    chunk_id: UUID
    ordinal: int
    text: str
    score: float
    char_start: int
    char_end: int
    # What the chunker knew about where this span came from, e.g.
    # `{"definition": "SyncSource._ingest"}`. Half of what makes the offsets
    # above usable as a citation rather than just a highlight range.
    metadata: dict[str, Any] = Field(default_factory=dict)


class HitOut(BaseModel):
    memory_id: UUID
    external_key: str
    # `(source_name, external_key)` is the durable identity of an item, and what a
    # judgement is recorded against. Returned so the client never has to guess it.
    source_name: str
    title: str | None
    kind: str
    occurred_at: datetime | None
    score: float
    # The evidence behind the score. Chunk-level provenance is exactly what
    # Phase 2's citations need, and it is available because M1.1 split these
    # tables rather than storing text on the memory.
    matched_chunks: list[ChunkOut]


class TimingOut(BaseModel):
    embed_ms: int
    search_ms: int
    total_ms: int


class SearchOut(BaseModel):
    query: str
    hits: list[HitOut]
    # Cheap now, and the only way to know where latency went once it matters.
    timing: TimingOut


class SearchIn(BaseModel):
    q: str = Field(min_length=1)
    k: int = 10
    # Several, because "compare these two connectors and nothing else" is a real
    # question and `SearchFilters` has always taken a list of ids. Repeat the
    # parameter to add sources: `?source=notes&source=code`. A single value still
    # works, so existing callers are unaffected.
    source: list[str] | None = None
    kind: MemoryKind | None = None
    after: datetime | None = None
    before: datetime | None = None
    include_deleted: bool = False
    ef_search: int | None = None
    exact: bool = False


async def build_filters(container: Container, body: SearchIn) -> SearchFilters:
    source_ids: list[UUID] | None = None
    if body.source:
        names = [name for name in body.source if name]
        async with container.database.session_factory() as session:
            found = {
                row[1]: row[0]
                for row in await session.execute(
                    select(models.Source.id, models.Source.name).where(
                        models.Source.name.in_(names)
                    )
                )
            }
        # Every name has to resolve. Quietly dropping an unknown one would return
        # results from the sources that *did* match and look like a successful
        # search of everything asked for.
        missing = [name for name in names if name not in found]
        if missing:
            raise HTTPException(
                status.HTTP_404_NOT_FOUND,
                f"no source named {', '.join(repr(name) for name in missing)}",
            )
        source_ids = [found[name] for name in names]

    return SearchFilters(
        source_ids=source_ids,
        kinds=[body.kind] if body.kind else None,
        occurred_after=body.after,
        occurred_before=body.before,
        include_deleted=body.include_deleted,
    )


def to_response(result: SearchResult) -> SearchOut:
    return SearchOut(
        query=result.query,
        timing=TimingOut(**result.timing.as_dict()),
        hits=[
            HitOut(
                memory_id=hit.memory_id,
                external_key=hit.external_key,
                source_name=hit.source_name,
                title=hit.title,
                kind=hit.kind.value,
                occurred_at=hit.occurred_at,  # type: ignore[arg-type]
                score=hit.score,
                matched_chunks=[
                    ChunkOut(
                        chunk_id=chunk.chunk_id,
                        ordinal=chunk.ordinal,
                        text=chunk.text,
                        score=chunk.score,
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        metadata=chunk.metadata,
                    )
                    for chunk in hit.matched_chunks
                ],
            )
            for hit in result.hits
        ],
    )


async def run_search(container: Container, body: SearchIn) -> SearchOut:
    filters = await build_filters(container, body)
    result = await container.search()(
        body.q,
        k=max(1, min(body.k, 100)),
        filters=filters,
        ef_search=body.ef_search,
        exact=body.exact,
    )
    return to_response(result)


@router.get("/search", response_model=SearchOut)
async def search(
    container: ContainerDep,
    q: Annotated[str, Query(min_length=1)],
    k: int = 10,
    source: Annotated[list[str] | None, Query()] = None,
    kind: MemoryKind | None = None,
    after: datetime | None = None,
    before: datetime | None = None,
    include_deleted: bool = False,
    ef_search: int | None = None,
    exact: bool = False,
) -> SearchOut:
    return await run_search(
        container,
        SearchIn(
            q=q,
            k=k,
            source=source,
            kind=kind,
            after=after,
            before=before,
            include_deleted=include_deleted,
            ef_search=ef_search,
            exact=exact,
        ),
    )


@router.post("/search", response_model=SearchOut)
async def search_post(container: ContainerDep, body: SearchIn) -> SearchOut:
    """Same thing, for queries too long to sit comfortably in a URL."""
    return await run_search(container, body)
