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
