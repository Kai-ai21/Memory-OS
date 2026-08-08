"""Source registration and sync triggering."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    DEFAULT_MAX_FILE_BYTES,
)
from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.container import Container
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobSpec, JobType
from memoryos.domain.values import SourceKind

router = APIRouter(tags=["sources"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class CreateSource(BaseModel):
    kind: SourceKind = SourceKind.FILESYSTEM
    name: str = Field(min_length=1)
    root: str
    include: list[str] | None = None
    exclude: list[str] | None = None
    max_file_bytes: int | None = None
    follow_symlinks: bool = False


class SourceOut(BaseModel):
    id: UUID
    kind: str
    name: str
    config: dict[str, Any]
    last_sync_at: datetime | None
    last_full_sync_at: datetime | None
    created_at: datetime


class SyncAccepted(BaseModel):
    job_id: UUID | None
    source_id: UUID
    full: bool


class MemoryOut(BaseModel):
    id: UUID
    source_id: UUID
    external_key: str
    version: int
    is_current: bool
    kind: str
    content_hash: str
    title: str | None
    occurred_at: datetime | None
    occurred_at_source: str
    ingested_at: datetime
    deleted_at: datetime | None


@router.post("/sources", response_model=SourceOut, status_code=status.HTTP_201_CREATED)
async def create_source(body: CreateSource, container: ContainerDep) -> models.Source:
    async with container.database.session_factory.begin() as session:
        repository = SqlAlchemySourceRepository(session)
        if await repository.get_by_name(body.kind, body.name) is not None:
            raise HTTPException(
                status.HTTP_409_CONFLICT, f"source {body.name!r} already exists"
            )

        source = Source(
            id=new_id(),
            kind=body.kind,
            name=body.name,
            config={
                "root": str(Path(body.root).expanduser().resolve()),
                "include": body.include or DEFAULT_INCLUDE,
                "exclude": body.exclude or DEFAULT_EXCLUDE,
                "max_file_bytes": body.max_file_bytes or DEFAULT_MAX_FILE_BYTES,
                "follow_symlinks": body.follow_symlinks,
            },
        )
        await repository.add(source)
        row = await session.get(models.Source, source.id)

    assert row is not None
    return row


@router.get("/sources", response_model=list[SourceOut])
async def list_sources(container: ContainerDep) -> list[models.Source]:
    async with container.database.session_factory() as session:
        result = await session.execute(select(models.Source).order_by(models.Source.name))
        return list(result.scalars())


@router.post(
    "/sources/{source_id}/sync",
    response_model=SyncAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def trigger_sync(
    source_id: UUID, container: ContainerDep, full: bool = False
) -> SyncAccepted:
    """Enqueue a sync. Never runs it inline.

    A sync of a large directory takes minutes. Running it in the request would
    blow the HTTP timeout, and whatever it managed to do would have no retry,
    no progress, and no way to resume. 202 with a job id is the honest answer.
    """
    async with container.database.session_factory.begin() as session:
        source = await session.get(models.Source, source_id)
        if source is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no such source")

        job_id = await enqueue_in(
            session,
            JobSpec(
                job_type=JobType.SYNC_SOURCE,
                payload={"source_id": str(source_id), "full": full},
                # One sync per source in flight at a time. A second request
                # while the first is still running returns its own 202 with a
                # null job id rather than queueing a duplicate walk.
                dedupe_key=f"{source_id}:{'full' if full else 'incremental'}",
            ),
        )

    return SyncAccepted(job_id=job_id, source_id=source_id, full=full)


@router.get("/memories", response_model=list[MemoryOut])
async def list_memories(
    container: ContainerDep,
    source_id: UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[models.Memory]:
    limit = max(1, min(limit, 500))
    stmt = (
        select(models.Memory)
        .where(models.Memory.is_current.is_(True))
        .order_by(models.Memory.ingested_at.desc(), models.Memory.external_key)
        .limit(limit)
        .offset(max(0, offset))
    )
    if source_id is not None:
        stmt = stmt.where(models.Memory.source_id == source_id)

    async with container.database.session_factory() as session:
        result = await session.execute(stmt)
        return list(result.scalars())
