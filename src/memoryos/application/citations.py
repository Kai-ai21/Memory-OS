"""Turning a ranked hit into something a reader can check.

The domain builds citations and explanations from numbers and text; this fetches
the text. One query for the memories behind the results, then arithmetic — and
it is skipped entirely when a caller asks for `explain=false`, because widening
an excerpt requires the parent memory's full normalized content and that is the
only part of a search that reads a large column.

The explanation is reconstructed here rather than recorded during fusion because
this is the layer that knows the weights *and* the final memory ordering. A
chunk's `rerank_rank` counts chunks in a shortlist; a result's position counts
memories. Comparing them is the mistake this module exists to not make.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import ScoredChunk
from memoryos.application.search import FusionWeights, MemoryHit
from memoryos.domain.citation import Citation, build_excerpt
from memoryos.domain.explanation import RankExplanation, build_explanation

logger = structlog.get_logger(__name__)

# How much text surrounds a quoted span by default. Two sentences either side,
# roughly: enough that "the second approach" has a first one, short enough that
# a result list stays scannable.
DEFAULT_CONTEXT_CHARS = 200


@dataclass(frozen=True, slots=True)
class ExplainedHit:
    """A hit with everything needed to justify it."""

    hit: MemoryHit
    citations: list[Citation]
    explanation: RankExplanation


async def explain_hits(
    session_factory: async_sessionmaker[AsyncSession],
    hits: Sequence[MemoryHit],
    *,
    weights: FusionWeights,
    rrf_k: int,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[ExplainedHit]:
    """Attach citations and an explanation to each hit, in rank order."""
    if not hits:
        return []

    content = await _memory_content(session_factory, [hit.memory_id for hit in hits])
    previous = _ranks_before_reranking(hits)

    explained: list[ExplainedHit] = []
    for rank, hit in enumerate(hits, start=1):
        text, version = content.get(hit.memory_id, (None, 1))
        explained.append(
            ExplainedHit(
                hit=hit,
                citations=[
                    _citation(hit, chunk, text, version, context_chars)
                    for chunk in hit.matched_chunks
                ],
                explanation=_explanation(hit, rank, previous.get(hit.memory_id), weights, rrf_k),
            )
        )
    return explained


def _citation(
    hit: MemoryHit,
    chunk: ScoredChunk,
    memory_text: str | None,
    version: int,
    context_chars: int,
) -> Citation:
    # The chunk's own span, with the overlap head borrowed from its predecessor
    # removed. This is the string `verify-citations` asserts equals
    # `memory.content[char_start:char_end]`.
    excerpt = chunk.text[chunk.prefix_chars :]
    definition = chunk.metadata.get("definition")

    return Citation(
        memory_id=hit.memory_id,
        source_name=hit.source_name,
        external_key=hit.external_key,
        chunk_ordinal=chunk.ordinal,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        prefix_chars=chunk.prefix_chars,
        excerpt=excerpt,
        definition=definition if isinstance(definition, str) else None,
        occurred_at=hit.occurred_at,  # type: ignore[arg-type]
        version=version,
        context=(
            None
            if memory_text is None
            else build_excerpt(
                memory_text,
                chunk.char_start,
                chunk.char_end,
                context_chars=context_chars,
            )
        ),
    )


def _explanation(
    hit: MemoryHit,
    rank: int,
    previous_rank: int | None,
    weights: FusionWeights,
    rrf_k: int,
) -> RankExplanation:
    """Explain the hit through its best chunk — the one that set its score.

    A memory scores as its best chunk, so that chunk is the reason it is here.
    Averaging the breakdowns of every matched chunk would describe a result
    nobody ranked.
    """
    best = max(hit.matched_chunks, key=lambda chunk: chunk.score, default=None)
    breakdown = best.breakdown if best is not None else None
    if breakdown is None:
        return build_explanation(
            final_rank=rank, fused_score=hit.score, ranks={}, rrf_k=rrf_k
        )

    ranks: dict[str, tuple[int, float | None, float]] = {}
    if breakdown.vector_rank is not None:
        ranks["semantic"] = (
            breakdown.vector_rank,
            breakdown.vector_score,
            weights.vector,
        )
    if breakdown.keyword_rank is not None:
        ranks["keyword"] = (
            breakdown.keyword_rank,
            breakdown.keyword_score,
            weights.keyword,
        )
    if breakdown.recency_rank is not None:
        ranks["recency"] = (
            breakdown.recency_rank,
            breakdown.recency_score,
            weights.recency,
        )
    if breakdown.importance_rank is not None:
        ranks["importance"] = (
            breakdown.importance_rank,
            breakdown.importance_score,
            weights.importance,
        )
    if breakdown.graph_rank is not None:
        ranks["graph"] = (
            breakdown.graph_rank,
            breakdown.graph_score,
            weights.graph,
        )

    return build_explanation(
        final_rank=rank,
        fused_score=breakdown.fused,
        ranks=ranks,
        rrf_k=rrf_k,
        rerank_score=breakdown.rerank_score,
        previous_rank=previous_rank,
        # The route, so the sentence can name the connection rather than only its
        # strength. Without it a graph-introduced result — one that may share no
        # word with the query — is explained as "weak graph match", which is a
        # number wearing a word. See `explanation._why`.
        graph_path=breakdown.graph_path,
    )


def _ranks_before_reranking(hits: Sequence[MemoryHit]) -> dict[UUID, int]:
    """Where each result would have placed on the fused score alone.

    Computed from the breakdowns already on the chunks rather than by running
    retrieval twice — the fused score survived reranking precisely so that this
    is arithmetic rather than a second search.

    Empty when nothing was reranked, so the explanation says nothing about a
    movement that did not happen.
    """
    if not any(
        chunk.breakdown is not None and chunk.breakdown.rerank_rank is not None
        for hit in hits
        for chunk in hit.matched_chunks
    ):
        return {}

    fused = [
        (
            hit.memory_id,
            max(
                (
                    chunk.breakdown.fused
                    for chunk in hit.matched_chunks
                    if chunk.breakdown is not None
                ),
                default=0.0,
            ),
        )
        for hit in hits
    ]
    ordered = sorted(fused, key=lambda pair: (-pair[1], str(pair[0])))
    return {memory_id: rank for rank, (memory_id, _) in enumerate(ordered, start=1)}


async def _memory_content(
    session_factory: async_sessionmaker[AsyncSession], memory_ids: Sequence[UUID]
) -> dict[UUID, tuple[str | None, int]]:
    stmt = select(models.Memory.id, models.Memory.content, models.Memory.version).where(
        models.Memory.id.in_(list(memory_ids))
    )
    async with session_factory() as session:
        return {row[0]: (row[1], row[2]) for row in await session.execute(stmt)}


# --------------------------------------------------------------------------
# Citations for things that are not search hits
# --------------------------------------------------------------------------
#
# **Everything above builds a citation from a `MemoryHit`, because until M7.0
# only search produced one.** Phase 7 makes every phase callable, and four of
# the six tools return something retrieval never touched: a graph neighbourhood,
# a range of dates, a silence, a decision. Each still has to be attributable —
# a tool result a model can read but not attribute is how the no-fabrication
# guardrail dies quietly — so the two functions below build the same `Citation`
# from a chunk or from a memory id.
#
# They do not widen the citation type or invent offsets for things that have
# none. A citation names a span of a version of a memory and `verify-citations`
# asserts that span still resolves; anything that cannot honestly produce one
# gets no citation and says so in its content instead.


@dataclass(frozen=True, slots=True)
class _Parent:
    source_name: str
    external_key: str
    version: int
    occurred_at: datetime | None
    content: str | None


async def _parents(
    session_factory: async_sessionmaker[AsyncSession], memory_ids: Sequence[UUID]
) -> dict[UUID, _Parent]:
    """Everything a citation needs about the memory a chunk belongs to."""
    if not memory_ids:
        return {}
    stmt = (
        select(
            models.Memory.id,
            models.Source.name,
            models.Memory.external_key,
            models.Memory.version,
            models.Memory.occurred_at,
            models.Memory.content,
        )
        .join(models.Source, models.Source.id == models.Memory.source_id)
        .where(models.Memory.id.in_(list(memory_ids)))
    )
    async with session_factory() as session:
        rows = await session.execute(stmt)
    return {
        row[0]: _Parent(
            source_name=row[1],
            external_key=row[2],
            version=row[3],
            occurred_at=row[4],
            content=row[5],
        )
        for row in rows
    }


async def citations_for_chunks(
    session_factory: async_sessionmaker[AsyncSession],
    chunks: Sequence[ScoredChunk],
    *,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[Citation]:
    """Citations for chunks that arrived by some route other than search.

    The graph expansion returns `ScoredChunk`s and no hit, and a memory read by
    id has chunk rows and no hit either. Both have the six fields a citation is
    made of; what they lack is the parent's name and version, which is one
    query.
    """
    parents = await _parents(session_factory, [chunk.memory_id for chunk in chunks])
    citations: list[Citation] = []
    for chunk in chunks:
        parent = parents.get(chunk.memory_id)
        if parent is None:
            # The memory was deleted between the ranking and this query. Skipped
            # rather than cited with a placeholder: an unresolvable citation is
            # worse than a missing one, because it looks checkable.
            continue
        excerpt = chunk.text[chunk.prefix_chars :]
        definition = chunk.metadata.get("definition")
        citations.append(
            Citation(
                memory_id=chunk.memory_id,
                source_name=parent.source_name,
                external_key=parent.external_key,
                chunk_ordinal=chunk.ordinal,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                prefix_chars=chunk.prefix_chars,
                excerpt=excerpt,
                definition=definition if isinstance(definition, str) else None,
                occurred_at=parent.occurred_at,
                version=parent.version,
                context=(
                    None
                    if parent.content is None
                    else build_excerpt(
                        parent.content,
                        chunk.char_start,
                        chunk.char_end,
                        context_chars=context_chars,
                    )
                ),
            )
        )
    return citations


async def citations_for_memories(
    session_factory: async_sessionmaker[AsyncSession],
    memory_ids: Sequence[UUID],
    *,
    context_chars: int = DEFAULT_CONTEXT_CHARS,
) -> list[Citation]:
    """One citation per memory, pointing at its opening span.

    **For results that are about memories rather than about passages** — a
    timeline bucket, a gap, the evidence under a decision. There is no matched
    span in any of those, so the first chunk is cited: it is a real span of a
    real version, it resolves, and it is where a reader opening the file would
    land. Citing the whole memory is not an option the type allows, and
    inventing `char_start=0, char_end=len(content)` would be a citation that
    passes `verify-citations` while pointing at everything.

    A memory with no chunks — not yet normalized — yields no citation. That is
    the honest answer rather than a citation to text that has not been split.
    """
    ordered = list(dict.fromkeys(memory_ids))
    if not ordered:
        return []
    stmt = (
        select(models.MemoryChunk)
        .where(
            models.MemoryChunk.memory_id.in_(ordered),
            models.MemoryChunk.ordinal == 0,
        )
    )
    async with session_factory() as session:
        rows = list((await session.execute(stmt)).scalars())

    by_memory = {row.memory_id: row for row in rows}
    chunks = [
        ScoredChunk(
            chunk_id=row.id,
            memory_id=row.memory_id,
            ordinal=row.ordinal,
            text=row.content,
            # A citation carries no score and this one has no ranking behind it.
            # Zero rather than a number that would imply one.
            score=0.0,
            char_start=row.char_start,
            char_end=row.char_end,
            prefix_chars=row.prefix_chars,
            metadata=dict(row.meta),
        )
        for memory_id in ordered
        if (row := by_memory.get(memory_id)) is not None
    ]
    return await citations_for_chunks(
        session_factory, chunks, context_chars=context_chars
    )


async def unresolved_locators(
    session_factory: async_sessionmaker[AsyncSession],
    citations: Sequence[Citation],
) -> list[str]:
    """The citations that no longer point at the text they claim to.

    **M2.5's identity, applied to the exact spans an answer rests on.** The
    property is the one `verify-citations` sweeps for corpus-wide:

        memory.content[char_start:char_end] == chunk.content[prefix_chars:]

    The difference is scope and timing. That command is a periodic sweep over
    every current chunk; this checks the handful a particular answer cited, at
    the moment the answer is produced, which is when it matters — an answer whose
    provenance has drifted is one whose quotations point somewhere else, and no
    amount of semantic support makes that acceptable.

    A citation to a memory that has since been deleted, or to a chunk ordinal
    that no longer exists, is unresolved rather than absent. Both are the corpus
    having moved under the answer, and both are reported by locator so a reader
    can see which claim lost its ground.
    """
    if not citations:
        return []

    wanted = {(citation.memory_id, citation.chunk_ordinal) for citation in citations}
    stmt = (
        select(
            models.MemoryChunk.memory_id,
            models.MemoryChunk.ordinal,
            models.MemoryChunk.content,
            models.MemoryChunk.char_start,
            models.MemoryChunk.char_end,
            models.MemoryChunk.prefix_chars,
            models.Memory.content,
        )
        .join(models.Memory, models.Memory.id == models.MemoryChunk.memory_id)
        .where(
            models.MemoryChunk.memory_id.in_({memory_id for memory_id, _ in wanted}),
            models.Memory.deleted_at.is_(None),
        )
    )
    async with session_factory() as session:
        rows = list(await session.execute(stmt))

    holds: set[tuple[UUID, int]] = set()
    for memory_id, ordinal, chunk_text, start, end, prefix, memory_text in rows:
        if memory_text is None:
            # No normalized text to check against. Not a failure — a memory can
            # legitimately be un-normalized — and not a pass either, so the pair
            # simply is not recorded as holding.
            continue
        if memory_text[start:end] == chunk_text[prefix:]:
            holds.add((memory_id, ordinal))

    unresolved = [
        citation.locator
        for citation in citations
        if (citation.memory_id, citation.chunk_ordinal) not in holds
    ]
    if unresolved:
        logger.warning(
            "citations.unresolved", count=len(unresolved), checked=len(citations)
        )
    return sorted(set(unresolved))
