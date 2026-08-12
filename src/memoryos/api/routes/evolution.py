"""One item's history over HTTP, with the diffs already computed.

**No model call happens on a GET unless it is asked for.** `summarize` defaults
to false and the endpoint returns whatever summaries are already cached, because
an endpoint the memory detail page hits on mount must not spend money per page
view — and a history of ten versions would be nine completions. Generation is an
explicit request.

The diffs are between consecutive versions by default, which is the history a
reader wants. `from`/`to` asks for one specific pair instead, which is what the
version selector sends: "what changed between v1 and v4" is a real question and
is not three consecutive diffs read together.
"""

from datetime import datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.adapters.llm.errors import MissingApiKey
from memoryos.application import evolution
from memoryos.container import Container
from memoryos.domain.jobs import PermanentError, TransientError

router = APIRouter(tags=["evolution"])

# Per-span text returned to a browser. A single span can be an entire inserted
# section — the README's second version added 2,506 characters in one — and the
# side-by-side view shows the change, not the file.
MAX_SPAN_CHARS = 4_000

# Spans per diff. A rewrite produces hundreds and nobody reads past the first
# screenful; the count is reported in full so the truncation is visible.
MAX_SPANS = 200


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class VersionOut(BaseModel):
    id: UUID
    version: int
    is_current: bool
    kind: str
    title: str | None
    content_hash: str
    normalized_hash: str | None
    occurred_at: datetime | None
    occurred_at_source: str
    ingested_at: datetime
    deleted_at: datetime | None
    characters: int
    chunks: int
    # False for every superseded version: M1.4 deletes an earlier version's
    # chunks when it writes the next one's. Sent so the client can render "0
    # chunks" as "discarded" rather than as a chunking result.
    holds_chunks: bool
    chunker_versions: list[str]
    adopted: bool | None
    text_changed: bool | None
    bytes_changed: bool | None
    change: str


class SpanOut(BaseModel):
    kind: str
    a_start: int
    a_end: int
    b_start: int
    b_end: int
    a_text: str
    b_text: str
    truncated: bool = False


class AffectedChunkOut(BaseModel):
    id: UUID
    ordinal: int
    char_start: int
    char_end: int
    definition: str | None
    spans: int


class SummaryOut(BaseModel):
    text: str
    model_id: str
    summarizer_version: str
    grounded: bool
    unsupported: list[str] = Field(default_factory=list)
    context_only: list[str] = Field(default_factory=list)
    trivial: bool
    cached: bool


class DiffOut(BaseModel):
    from_id: UUID
    to_id: UUID
    from_version: int
    to_version: int
    added_chars: int
    removed_chars: int
    # None when either side no longer holds its chunks, which is every pair
    # involving a superseded version. Not zero, and not the newer side's count.
    chunk_delta: int | None
    is_empty: bool
    span_count: int
    spans: list[SpanOut]
    affected_chunks: list[AffectedChunkOut]
    unified: str
    summary: SummaryOut | None = None


class EvolutionOut(BaseModel):
    memory_id: UUID
    source_id: UUID
    source_name: str
    external_key: str
    versions: list[VersionOut]
    diffs: list[DiffOut]


@router.get("/memories/{memory_id}/evolution", response_model=EvolutionOut)
async def get_evolution(
    memory_id: UUID,
    container: ContainerDep,
    from_id: Annotated[
        UUID | None, Query(alias="from", description="Diff this version instead.")
    ] = None,
    to_id: Annotated[UUID | None, Query(alias="to", description="…against this one.")] = None,
    summarize: Annotated[
        bool,
        Query(description="Generate missing summaries. Costs a model call per diff."),
    ] = False,
) -> EvolutionOut:
    sessions = container.database.session_factory
    async with sessions() as session:
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

    history = await evolution.version_history(sessions, memory.source_id, memory.external_key)

    if (from_id is None) != (to_id is None):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "'from' and 'to' must be given together",
        )
    pairs = (
        [(from_id, to_id)]
        if from_id is not None and to_id is not None
        else [
            (history[index - 1].id, history[index].id) for index in range(1, len(history))
        ]
    )

    summarizer = None
    if summarize:
        try:
            summarizer = evolution.SummarizeChange(sessions, container.language_model())
        except MissingApiKey as exc:
            # A missing key is a configuration fact, not a server fault, and the
            # history is still worth returning without it.
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    diffs: list[DiffOut] = []
    for before_id, after_id in pairs:
        try:
            diff = await evolution.diff_versions(sessions, before_id, after_id)
        except PermanentError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        diffs.append(await _to_diff(diff, summarizer))

    return EvolutionOut(
        memory_id=memory.id,
        source_id=memory.source_id,
        source_name=source_name,
        external_key=memory.external_key,
        versions=[_to_version(version) for version in history],
        diffs=diffs,
    )


async def _to_diff(
    diff: evolution.VersionDiff, summarizer: evolution.SummarizeChange | None
) -> DiffOut:
    summary = None
    if summarizer is not None:
        try:
            produced = await summarizer(diff)
        except (TransientError, PermanentError) as exc:
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, f"the language model could not answer: {exc}"
            ) from exc
        summary = SummaryOut(
            text=produced.text,
            model_id=produced.model_id,
            summarizer_version=produced.summarizer_version,
            grounded=produced.grounding.grounded,
            unsupported=list(produced.grounding.unsupported),
            context_only=list(produced.grounding.context_only),
            trivial=produced.trivial,
            cached=produced.cached,
        )

    return DiffOut(
        from_id=diff.before.id,
        to_id=diff.after.id,
        from_version=diff.before.version,
        to_version=diff.after.version,
        added_chars=diff.added_chars,
        removed_chars=diff.removed_chars,
        chunk_delta=diff.chunk_delta,
        is_empty=diff.is_empty,
        # The real count, beside a truncated list, so a client showing 200 spans
        # can say how many it is not showing.
        span_count=len(diff.spans),
        spans=[
            SpanOut(
                kind=span.kind.value,
                a_start=span.a_start,
                a_end=span.a_end,
                b_start=span.b_start,
                b_end=span.b_end,
                a_text=span.a_text[:MAX_SPAN_CHARS],
                b_text=span.b_text[:MAX_SPAN_CHARS],
                truncated=max(len(span.a_text), len(span.b_text)) > MAX_SPAN_CHARS,
            )
            for span in diff.spans[:MAX_SPANS]
        ],
        affected_chunks=[
            AffectedChunkOut(
                id=chunk.id,
                ordinal=chunk.ordinal,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                definition=chunk.definition,
                spans=chunk.spans,
            )
            for chunk in diff.affected_chunks
        ],
        unified=diff.unified_diff,
        summary=summary,
    )


def _to_version(version: evolution.MemoryVersion) -> VersionOut:
    return VersionOut(
        id=version.id,
        version=version.version,
        is_current=version.is_current,
        kind=version.kind,
        title=version.title,
        content_hash=version.content_hash,
        normalized_hash=version.normalized_hash,
        occurred_at=version.occurred_at,
        occurred_at_source=version.occurred_at_source,
        ingested_at=version.ingested_at,
        deleted_at=version.deleted_at,
        characters=version.characters,
        chunks=version.chunks,
        holds_chunks=version.holds_chunks,
        chunker_versions=list(version.chunker_versions),
        adopted=version.adopted,
        text_changed=version.text_changed,
        bytes_changed=version.bytes_changed,
        change=version.summary_of_change,
    )
