"""Semantic search over memories."""

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field
from sqlalchemy import select

from memoryos.adapters.db import models
from memoryos.application.citations import ExplainedHit, explain_hits
from memoryos.application.ports import SearchFilters
from memoryos.application.search import SearchResult
from memoryos.container import Container
from memoryos.domain.citation import Citation
from memoryos.domain.fusion import DEFAULT_RRF_K
from memoryos.domain.values import DEFAULT_SEARCH_MODE, MemoryKind, SearchMode

router = APIRouter(tags=["search"])


def get_container(request: Request) -> Container:
    container: Container = request.app.state.container
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


class BreakdownOut(BaseModel):
    """Where the score came from, per retriever.

    Returned on every hit in every mode. Once two retrievers are fused into one
    number, that number is the only thing a client can show, and without this it
    cannot say *why* — which is both how a bad ranking gets debugged and what
    M2.5's citations will render. A null rank means that retriever did not
    return the chunk, which is not the same as returning it last.
    """

    fused: float
    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_score: float | None = None
    keyword_score: float | None = None
    recency_rank: int | None = None
    importance_rank: int | None = None
    recency_score: float | None = None
    importance_score: float | None = None
    # Null means the cross-encoder never saw this chunk — it fell outside the
    # shortlist, or reranking was off. Not the same as scoring badly.
    rerank_score: float | None = None
    rerank_rank: int | None = None
    # M3.5's graph expansion. `graph_path` is the entity route that reached this
    # chunk, rendered as `job queue -> SKIP LOCKED`, and it is the reason the
    # client can show *why* a result no retriever found is in the list. A rank
    # without a route would be the graph asserting something unfalsifiable, which
    # is exactly what M2.5 built this object to prevent.
    graph_rank: int | None = None
    graph_score: float | None = None
    graph_path: str | None = None


class ExcerptOut(BaseModel):
    """A quotable window with the matched span located inside it.

    `span_start`/`span_end` index into `text`, not into the memory, so a client
    highlights without redoing the offset arithmetic — which is exactly the
    arithmetic that gets it wrong.
    """

    text: str
    span_start: int
    span_end: int
    truncated_start: bool
    truncated_end: bool


class CitationOut(BaseModel):
    memory_id: UUID
    source_name: str
    external_key: str
    chunk_ordinal: int
    char_start: int
    char_end: int
    prefix_chars: int
    # Exactly `memory.content[char_start:char_end]`, with the overlap head a
    # chunk borrows from its predecessor removed. `verify-citations` asserts it.
    excerpt: str
    definition: str | None = None
    occurred_at: datetime | None = None
    # Which version was quoted. A citation to a memory that has since changed
    # has to say what it referred to.
    version: int
    context: ExcerptOut | None = None


class ContributionOut(BaseModel):
    name: str
    rank: int
    score: float | None = None
    weight: float
    contribution: float
    # This ranking's percentage of the fused score. The one number that answers
    # "why is this third?".
    share: float


class ExplanationOut(BaseModel):
    final_rank: int
    fused_score: float
    contributions: list[ContributionOut]
    rerank_score: float | None = None
    # The entity route, when the graph is what put this result here. Carried
    # alongside `why`, which already names it, so a client can render the path as
    # something clickable rather than parsing it back out of a sentence.
    graph_path: str | None = None
    # Assembled from the numbers above, never generated. Deterministic, free,
    # and available on every result.
    why: str


class ChunkOut(BaseModel):
    chunk_id: UUID
    ordinal: int
    text: str
    score: float
    char_start: int
    char_end: int
    breakdown: BreakdownOut | None = None
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
    # Beside the date it qualifies. An mtime and a date an email declared are
    # different claims, and a client that received only the timestamps could not
    # render them differently however much it wanted to.
    occurred_at_source: str
    score: float
    # The evidence behind the score. Chunk-level provenance is exactly what
    # Phase 2's citations need, and it is available because M1.1 split these
    # tables rather than storing text on the memory.
    matched_chunks: list[ChunkOut]
    # Present unless the caller passed `explain=false`. Both require reading the
    # parent memory's full normalized text, which is the only large column a
    # search touches.
    citations: list[CitationOut] | None = None
    explanation: ExplanationOut | None = None


class TimingOut(BaseModel):
    embed_ms: int
    search_ms: int
    rerank_ms: int
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
    # Hybrid: both retrievers, fused by RRF. `vector` and `keyword` remain
    # reachable because the only way to know what fusion is doing is to be able
    # to run each half alone. `ef_search` and `exact` are vector-only knobs and
    # do nothing under `keyword`.
    mode: SearchMode = DEFAULT_SEARCH_MODE
    # The expensive half. False returns the fused ordering, which is what makes
    # the reranker's contribution measurable rather than assumed.
    rerank: bool = True
    # Citations and the ranking explanation. Both cost one extra query for the
    # memories' normalized text; a caller that only needs the ranking can skip it.
    explain: bool = True


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


def to_response(
    result: SearchResult, explained: list[ExplainedHit] | None = None
) -> SearchOut:
    extras = {item.hit.memory_id: item for item in (explained or [])}
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
                occurred_at_source=hit.occurred_at_source.value,
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
                        breakdown=(
                            None
                            if chunk.breakdown is None
                            else BreakdownOut(**chunk.breakdown.as_dict())  # type: ignore[arg-type]
                        ),
                    )
                    for chunk in hit.matched_chunks
                ],
                citations=(
                    None
                    if hit.memory_id not in extras
                    else [
                        _citation_out(citation)
                        for citation in extras[hit.memory_id].citations
                    ]
                ),
                explanation=(
                    None
                    if hit.memory_id not in extras
                    else ExplanationOut(
                        **extras[hit.memory_id].explanation.as_dict()  # type: ignore[arg-type]
                    )
                ),
            )
            for hit in result.hits
        ],
    )


def _citation_out(citation: Citation) -> CitationOut:
    return CitationOut(
        memory_id=citation.memory_id,
        source_name=citation.source_name,
        external_key=citation.external_key,
        chunk_ordinal=citation.chunk_ordinal,
        char_start=citation.char_start,
        char_end=citation.char_end,
        prefix_chars=citation.prefix_chars,
        excerpt=citation.excerpt,
        definition=citation.definition,
        occurred_at=citation.occurred_at,
        version=citation.version,
        context=(
            None
            if citation.context is None
            else ExcerptOut(
                text=citation.context.text,
                span_start=citation.context.span_start,
                span_end=citation.context.span_end,
                truncated_start=citation.context.truncated_start,
                truncated_end=citation.context.truncated_end,
            )
        ),
    )


async def run_search(container: Container, body: SearchIn) -> SearchOut:
    filters = await build_filters(container, body)
    result = await container.search()(
        body.q,
        k=max(1, min(body.k, 100)),
        filters=filters,
        ef_search=body.ef_search,
        exact=body.exact,
        mode=body.mode,
        rerank=body.rerank,
    )
    explained = None
    if body.explain:
        explained = await explain_hits(
            container.database.session_factory,
            result.hits,
            weights=container.weights(),
            rrf_k=DEFAULT_RRF_K,
        )
    return to_response(result, explained)


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
    mode: SearchMode = DEFAULT_SEARCH_MODE,
    rerank: bool = True,
    explain: bool = True,
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
            mode=mode,
            rerank=rerank,
            explain=explain,
        ),
    )


@router.post("/search", response_model=SearchOut)
async def search_post(container: ContainerDep, body: SearchIn) -> SearchOut:
    """Same thing, for queries too long to sit comfortably in a URL."""
    return await run_search(container, body)
