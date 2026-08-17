"""Source registration and sync triggering."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from memoryos.adapters.connectors.filesystem import (
    DEFAULT_EXCLUDE,
    DEFAULT_INCLUDE,
    DEFAULT_MAX_FILE_BYTES,
)
from memoryos.adapters.db import models
from memoryos.adapters.db.job_queue import enqueue_in
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.api.routes.chat import LOG_NOTE
from memoryos.application import deletion, export
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
    # Current, undeleted memories and their chunks. Zero is the interesting
    # value: a registered source with no memories has never synced or matched
    # nothing, and a total corpus count cannot tell you which source that is.
    memories: int = 0
    chunks: int = 0


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
async def list_sources(container: ContainerDep) -> list[SourceOut]:
    """Every source, with how much of the corpus came from it.

    Outer-joined and grouped rather than a count per source in a loop: the list
    is short today, but a query per row is the shape that stops being fine
    without anybody noticing.
    """
    stmt = (
        select(
            models.Source,
            func.count(func.distinct(models.Memory.id)),
            func.count(models.MemoryChunk.id),
        )
        .outerjoin(
            models.Memory,
            (models.Memory.source_id == models.Source.id)
            & models.Memory.is_current.is_(True)
            & models.Memory.deleted_at.is_(None),
        )
        .outerjoin(models.MemoryChunk, models.MemoryChunk.memory_id == models.Memory.id)
        .group_by(models.Source.id)
        .order_by(models.Source.name)
    )
    async with container.database.session_factory() as session:
        rows = (await session.execute(stmt)).all()

    return [
        SourceOut(
            id=source.id,
            kind=source.kind,
            name=source.name,
            config=source.config,
            last_sync_at=source.last_sync_at,
            last_full_sync_at=source.last_full_sync_at,
            created_at=source.created_at,
            memories=memories,
            chunks=chunks,
        )
        for source, memories, chunks in rows
    ]


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


class SourceDeletionScopeOut(BaseModel):
    """What deleting a whole source would remove.

    **The most destructive operation in the product, so the counts are exact rather
    than approximate and they are read at the moment of asking.** A dialog that said
    "this will delete a lot" is not a dialog somebody can consent to, and one that
    reused a count from the sources list would be confirming against whatever was
    true when the page loaded.
    """

    source: str
    items: int
    memories: int
    chunks: int
    embedded_chunks: int
    mentions: int
    orphaned_entities: int
    tags: int
    turns: int
    attachments: int
    evidence: int
    blobs: int
    shared_blobs: int
    previews: list[str]
    log_note: str


class SourceDeletionOut(BaseModel):
    source: str
    items: int
    memories: int
    chunks: int
    mentions: int
    tags: int
    turns: int
    blobs_shredded: int
    # False when the registration survived because the log still references it.
    # Reported rather than assumed: "the source is gone" and "the source is empty
    # and its config is still here" are different states, and the second is what
    # actually happens for any source that ever observed anything.
    source_removed: bool
    detail: str


@router.get(
    "/sources/{source_id}/deletion", response_model=SourceDeletionScopeOut
)
async def source_deletion_scope(
    source_id: UUID, container: ContainerDep
) -> SourceDeletionScopeOut:
    """Exactly what deleting this source would take."""
    try:
        scope = await deletion.scope_of_source(
            container.database.session_factory, source_id
        )
    except deletion.NoSuchMemory as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    name = scope.items[0].source_name if scope.items else await _source_name(
        container, source_id
    )
    return SourceDeletionScopeOut(
        source=name,
        items=len(scope.items),
        memories=scope.memories,
        chunks=scope.chunks,
        embedded_chunks=scope.embedded_chunks,
        mentions=scope.mentions,
        orphaned_entities=scope.orphaned_entities,
        tags=scope.tags,
        turns=scope.turns,
        attachments=scope.attachments,
        evidence=scope.evidence,
        blobs=scope.blobs,
        shared_blobs=scope.shared_blobs,
        previews=list(scope.previews),
        log_note=LOG_NOTE,
    )


@router.delete("/sources/{source_id}", response_model=SourceDeletionOut)
async def delete_source(
    source_id: UUID,
    container: ContainerDep,
    confirm_items: Annotated[int | None, Query()] = None,
    keep_registration: Annotated[bool, Query()] = False,
) -> SourceDeletionOut:
    """Permanently delete everything a source produced.

    **`confirm_items` is a second key on the launch panel, and it is required by
    convention rather than by the signature.** A client is expected to read
    `/sources/{id}/deletion`, show the counts, and send back the item count it
    showed. If the corpus has changed since — a sync landed, another tab deleted
    something — the numbers disagree and this refuses, because the person consented
    to a specific amount of loss and the operation is now a different one.

    Omitting it is allowed, deliberately: this is a local-first tool and `curl`
    without a ceremony parameter is a legitimate way to use it. What is not allowed
    is a *wrong* count, which is the case that means somebody confirmed something
    else.
    """
    name = await _source_name(container, source_id)
    scope = await deletion.scope_of_source(container.database.session_factory, source_id)
    if confirm_items is not None and confirm_items != len(scope.items):
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"the confirmation names {confirm_items} item(s) and this source now "
            f"has {len(scope.items)}. The corpus changed since the count was read, "
            f"so nothing has been deleted — read /sources/{source_id}/deletion "
            f"again and confirm the current numbers.",
        )

    try:
        report = await deletion.purge_source(
            container.database.session_factory,
            container.blobs,
            source_id,
            drop_source=not keep_registration,
        )
    except deletion.NoSuchMemory as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except deletion.BlobsSurvived as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc

    async with container.database.session_factory() as session:
        still_there = await session.get(models.Source, source_id)

    return SourceDeletionOut(
        source=name,
        items=report.items,
        memories=report.memories,
        chunks=report.chunks,
        mentions=report.mentions,
        tags=report.tags,
        turns=report.turns,
        blobs_shredded=report.blobs_shredded,
        source_removed=still_there is None,
        detail=(
            f"Permanently deleted {report.items} item(s) from {name!r}: "
            f"{report.memories} version(s), {report.chunks} chunk(s) and their "
            f"vectors, {report.mentions} entity mention(s), {report.turns} "
            f"transcript row(s) and {report.blobs_shredded} stored file(s). "
            + (
                "The source registration is gone too."
                if still_there is None
                else "The source registration and its config are kept, because the "
                "ingestion log still records what it observed."
            )
        ),
    )


class ReindexAccepted(BaseModel):
    source: str
    memories: int
    jobs: int


@router.post(
    "/sources/{source_id}/reindex",
    response_model=ReindexAccepted,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex_source(source_id: UUID, container: ContainerDep) -> ReindexAccepted:
    """Re-run the pipeline over everything this source holds.

    **Re-normalization, not re-ingestion, and the difference is what makes this
    cheap and safe.** Nothing is re-read from the source and no event is appended:
    the artifacts are already in the blob store and the log already says what was
    observed. This enqueues one `NORMALIZE_MEMORY` job per current memory, and the
    existing pipeline then re-parses, re-chunks and re-embeds — which is the
    operation somebody wants after a chunker change or a parser fix, and it is the
    same work `rechunk` does for a subset.

    202 with a count, never inline. A corpus-sized re-index is minutes of model
    time.

    Deliberately does not clear `entity_extractor_version`. Re-extraction is real
    money per chunk and is its own command; a re-index that silently spent it would
    be a button whose cost is invisible.
    """
    name = await _source_name(container, source_id)
    async with container.database.session_factory.begin() as session:
        memory_ids = (
            await session.execute(
                select(models.Memory.id).where(
                    models.Memory.source_id == source_id,
                    models.Memory.is_current.is_(True),
                    # Tombstoned items are skipped. Re-indexing something removed
                    # from view would put its chunks back into the vector index,
                    # which is the deletion guardrail being undone by a maintenance
                    # command.
                    models.Memory.deleted_at.is_(None),
                )
            )
        ).scalars().all()

        queued = 0
        for memory_id in memory_ids:
            job_id = await enqueue_in(
                session,
                JobSpec(
                    job_type=JobType.NORMALIZE_MEMORY,
                    payload={"memory_id": str(memory_id)},
                    # The same dedupe key `ingest_item` uses, so a re-index during
                    # an active sync cannot queue a second normalization of a
                    # memory whose first one is still pending.
                    dedupe_key=str(memory_id),
                ),
            )
            queued += int(job_id is not None)

    return ReindexAccepted(source=name, memories=len(memory_ids), jobs=queued)


@router.get("/sources/{source_id}/export")
async def export_source(source_id: UUID, container: ContainerDep) -> StreamingResponse:
    """Everything from one source, as JSON, with every version.

    Streamed rather than assembled. A source can hold a corpus, and the format is
    the one `memoryos export` writes — one implementation, so a file downloaded from
    the browser and a file written by the CLI cannot differ.
    """
    name = await _source_name(container, source_id)
    return StreamingResponse(
        export.to_json(container.database.session_factory, source=name),
        media_type="application/json",
        headers={
            # Named after the source, so a directory of these is readable. Quoted
            # because a source name may contain a space.
            "Content-Disposition": f'attachment; filename="memoryos-{name}.json"'
        },
    )


async def _source_name(container: Container, source_id: UUID) -> str:
    async with container.database.session_factory() as session:
        row = await session.get(models.Source, source_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such source")
    return row.name


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
