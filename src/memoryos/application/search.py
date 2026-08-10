"""Search: chunks are what match, memories are what you get back.

Two retrievers and one fusion, behind one use case. `mode` chooses; hybrid is
the default and the product. Everything downstream of the retrieval step is
shared — the same grouping into memories, the same max-scoring, the same
response shape — which is what M2.1 was shaped for and what makes hybrid an
extra step here rather than a second pipeline.

The fused score replaces the chunk's score, because everything downstream ranks
on `ScoredChunk.score` and a mixture of cosine similarities and RRF scores in
one list would sort into nonsense. The retriever's own score is not lost: it
moves to `ScoreBreakdown`, which is the only reason a fused ranking can be
argued with afterwards.
"""

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from statistics import mean
from uuid import UUID

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.db import models
from memoryos.application.ports import (
    Embedder,
    KeywordStore,
    ScoreBreakdown,
    ScoredChunk,
    SearchFilters,
    VectorStore,
)
from memoryos.domain.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from memoryos.domain.signals import recency_score
from memoryos.domain.values import DEFAULT_SEARCH_MODE, MemoryKind, SearchMode

logger = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(UTC)

# Chunks fetched per requested memory. Several chunks of one document commonly
# match together, so retrieving only k would routinely yield fewer than k
# distinct memories.
CHUNK_FANOUT = 5

# Fusion depth, per retriever. RRF can only reward agreement it can see: a chunk
# at rank 15 in one retriever and rank 2 in the other has to be present in both
# lists to be fused at all, and truncating either at k discards exactly the
# disagreements fusion exists to resolve.
#
# This is a floor, not the number used — see `_hybrid`. Each leg fetches
# `max(k * FUSION_FANOUT, k * CHUNK_FANOUT)`, because the two constraints are
# independent and both bind:
#
#   * fusion needs depth to compare rankings, which is this constant;
#   * grouping needs ~5 chunks per memory to yield k memories, which is
#     CHUNK_FANOUT, and that requirement does not go away because a second
#     retriever arrived.
#
# Taking only the first left the vector leg shallower inside hybrid than the
# vector-only path, and it cost real recall rather than a rounding error:
# measured on the golden set, 3 gives recall 0.799 and 5 gives 0.815, because
# `entities.py` chunk 4 — the pinned answer to one of the queries — sits in the
# band between 30 and 50 chunks and simply was not in the list to be fused.
# A hybrid handicapped against its own vector half measures fanout, not fusion.
FUSION_FANOUT = 3


@dataclass(frozen=True, slots=True)
class MemoryHit:
    memory_id: UUID
    external_key: str
    # Which connector this came from. Carried because `(source_name,
    # external_key)` is the durable identity of an item — the pair a judgement is
    # recorded against, and the pair `verify-replay` compares on. A caller that
    # had to infer it would be guessing, and `external_key` alone is not unique
    # across sources.
    source_name: str
    title: str | None
    kind: MemoryKind
    occurred_at: object
    score: float
    matched_chunks: list[ScoredChunk]


@dataclass(slots=True)
class SearchTiming:
    embed_ms: int = 0
    search_ms: int = 0
    total_ms: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "embed_ms": self.embed_ms,
            "search_ms": self.search_ms,
            "total_ms": self.total_ms,
        }


@dataclass(slots=True)
class SearchResult:
    query: str
    hits: list[MemoryHit] = field(default_factory=list)
    timing: SearchTiming = field(default_factory=SearchTiming)
    mode: SearchMode = SearchMode.VECTOR


@dataclass(frozen=True, slots=True)
class FusionWeights:
    """How much each ranking counts. Defaults match `config.Settings`.

    Signals default to zero because the grid search said so, not because they
    are unimplemented — see the settings comment for the numbers.
    """

    vector: float = 1.0
    keyword: float = 1.0
    recency: float = 0.0
    importance: float = 0.0

    @property
    def signals_are_off(self) -> bool:
        """True when this is M2.2 hybrid exactly, with no signal contribution."""
        return self.recency == 0.0 and self.importance == 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "vector": self.vector,
            "keyword": self.keyword,
            "recency": self.recency,
            "importance": self.importance,
        }


@dataclass(frozen=True, slots=True)
class SignalTable:
    """Per-chunk signal scores for one query's candidates.

    **Signals rank candidates; they never introduce them.** The two rankings
    below cover exactly the chunks the retrievers already found, so a file can
    be promoted for being recent but cannot appear for it. Ranking the whole
    corpus by date and fusing that in would surface last week's untouched notes
    for every query, which is not a ranking signal but a different product.

    Ordering by `recency_score` descending is identical to ordering by
    `occurred_at` descending, so the recency *ranking* does not change as the
    clock moves even though the scores do. The one exception is an undated
    chunk, pinned at 0.5: dated items decay past it over months, so its position
    drifts slowly. That is the intended meaning of "one half-life old".
    """

    recency: dict[str, float] = field(default_factory=dict)
    importance: dict[str, float] = field(default_factory=dict)

    def ranking(self, scores: dict[str, float]) -> list[str]:
        """Chunk ids best first, ties broken by id so a rerun agrees."""
        return [key for key, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


class SearchMemories:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: Embedder,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        weights: FusionWeights | None = None,
        *,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        self._sessions = session_factory
        self._embedder = embedder
        self._store = vector_store
        self._keywords = keyword_store
        self._weights = weights or FusionWeights()
        # Injected so a test can pin the clock. Recency ranks by `occurred_at`
        # order, which is invariant under a changing `now` — see `SignalTable` —
        # so this affects the reported scores rather than the ranking.
        self._now = now

    async def __call__(
        self,
        query: str,
        *,
        k: int = 10,
        filters: SearchFilters | None = None,
        ef_search: int | None = None,
        exact: bool = False,
        mode: SearchMode = DEFAULT_SEARCH_MODE,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> SearchResult:
        started = time.monotonic()
        resolved = filters or SearchFilters()

        embed_ms = 0
        search_started = time.monotonic()
        if mode is SearchMode.HYBRID:
            chunks, embed_ms, search_started = await self._hybrid(
                query, k=k, filters=resolved, ef_search=ef_search, rrf_k=rrf_k
            )
        elif mode is SearchMode.KEYWORD:
            # No embedding at all, which is most of why this mode is ~200x
            # faster on a cold process: the model never loads.
            chunks = _attribute_single(
                await self._keywords.search(
                    query, k=max(k * CHUNK_FANOUT, k), filters=resolved
                ),
                mode,
            )
        else:
            embed_started = time.monotonic()
            vector = await self._embed(query)
            embed_ms = _elapsed_ms(embed_started)

            search_started = time.monotonic()
            chunks = _attribute_single(
                await self._vector_chunks(
                    vector,
                    k=max(k * CHUNK_FANOUT, k),
                    filters=resolved,
                    ef_search=ef_search,
                    exact=exact,
                ),
                mode,
            )
        search_ms = _elapsed_ms(search_started)

        hits = await self._to_hits(chunks, k=k)

        timing = SearchTiming(
            embed_ms=embed_ms, search_ms=search_ms, total_ms=_elapsed_ms(started)
        )
        logger.info(
            "search.finished",
            query_length=len(query),
            chunks=len(chunks),
            hits=len(hits),
            mode=mode.value,
            # Only meaningful in vector mode; there is no approximate index to
            # bypass on the keyword side.
            exact=exact and mode is SearchMode.VECTOR,
            **timing.as_dict(),
        )
        return SearchResult(query=query, hits=hits, timing=timing, mode=mode)

    async def _hybrid(
        self,
        query: str,
        *,
        k: int,
        filters: SearchFilters,
        ef_search: int | None,
        rrf_k: int,
    ) -> tuple[list[ScoredChunk], int, float]:
        """Both retrievers, concurrently, fused into one ranking.

        Returns the chunks along with the embedding cost and the moment
        retrieval began, because under concurrency those two overlap and the
        caller cannot reconstruct them.

        The two legs are independent queries against the same database and
        running them in sequence would add the embedding latency to the keyword
        latency for no reason at all. `gather` also means the keyword query —
        which needs no model — is already in flight while the embedder works.
        """
        wanted = max(k * FUSION_FANOUT, k * CHUNK_FANOUT, k)

        embed_started = time.monotonic()

        async def vector_leg() -> tuple[list[ScoredChunk], int]:
            vector = await self._embed(query)
            elapsed = _elapsed_ms(embed_started)
            found = await self._vector_chunks(
                vector, k=wanted, filters=filters, ef_search=ef_search, exact=False
            )
            return found, elapsed

        (vector_chunks, embed_ms), keyword_chunks = await asyncio.gather(
            vector_leg(),
            self._keywords.search(query, k=wanted, filters=filters),
        )

        signals = await self.signals_for(vector_chunks, keyword_chunks)

        return (
            fuse(
                vector_chunks,
                keyword_chunks,
                signals=signals,
                weights=self._weights,
                rrf_k=rrf_k,
            ),
            embed_ms,
            embed_started,
        )

    async def vector_candidates(
        self, query: str, *, k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        """The vector leg alone. For the tuner, which re-fuses without re-retrieving."""
        return await self._vector_chunks(
            await self._embed(query), k=k, filters=filters, ef_search=None, exact=False
        )

    async def keyword_candidates(
        self, query: str, *, k: int, filters: SearchFilters
    ) -> list[ScoredChunk]:
        """The keyword leg alone. Counterpart to `vector_candidates`."""
        return await self._keywords.search(query, k=k, filters=filters)

    async def memory_metadata(
        self, memory_ids: list[UUID]
    ) -> dict[UUID, tuple[str, str | None, str, object, str]]:
        """Public because the tuner groups chunks into memories without re-searching."""
        return await self._memory_metadata(memory_ids)

    async def signals_for(
        self, *chunk_lists: Sequence[ScoredChunk]
    ) -> SignalTable:
        """Signal scores for every candidate chunk, keyed by chunk id.

        One query for the memories behind the candidates, then arithmetic. The
        chunk inherits its memory's recency and importance: a signal about *when
        a document was written* has no per-chunk refinement to offer, and
        pretending otherwise would invent precision.
        """
        candidates = {
            str(chunk.chunk_id): chunk.memory_id
            for chunks in chunk_lists
            for chunk in chunks
        }
        if not candidates:
            return SignalTable()

        stmt = select(
            models.Memory.id, models.Memory.occurred_at, models.Memory.importance
        ).where(models.Memory.id.in_(set(candidates.values())))
        async with self._sessions() as session:
            rows = {row[0]: (row[1], row[2]) for row in await session.execute(stmt)}

        now = self._now()
        recency: dict[str, float] = {}
        importance: dict[str, float] = {}
        for chunk_id, memory_id in candidates.items():
            row = rows.get(memory_id)
            if row is None:
                continue
            occurred_at, weight = row
            recency[chunk_id] = recency_score(occurred_at, now)
            # A null column means `recompute-importance` has not run. Skipped
            # rather than defaulted: scoring it 0.0 would rank every unscored
            # memory last, and 0.5 would invent a middle the data has not
            # earned. An absent entry simply does not appear in the ranking.
            if weight is not None:
                importance[chunk_id] = float(weight)

        return SignalTable(recency=recency, importance=importance)

    async def _embed(self, query: str) -> list[float]:
        # Same reasoning as the embed pipeline: CPU-bound matrix work does not
        # belong on the event loop, even for a single short query.
        #
        # `embed_query`, not `embed_passage`: this text is what the caller is
        # searching *with*, and the model was trained to see the two differently.
        (vector,) = await asyncio.to_thread(self._embedder.embed_query, [query])
        return vector

    async def _vector_chunks(
        self,
        vector: list[float],
        *,
        k: int,
        filters: SearchFilters,
        ef_search: int | None,
        exact: bool,
    ) -> list[ScoredChunk]:
        if exact:
            return await self._store.search_exact(vector, k=k, filters=filters)
        return await self._store.search(vector, k=k, filters=filters, ef_search=ef_search)

    async def _to_hits(self, chunks: list[ScoredChunk], *, k: int) -> list[MemoryHit]:
        if not chunks:
            return []

        grouped: dict[UUID, list[ScoredChunk]] = {}
        for chunk in chunks:
            grouped.setdefault(chunk.memory_id, []).append(chunk)

        metadata = await self._memory_metadata(list(grouped))

        hits: list[MemoryHit] = []
        for memory_id, matched in grouped.items():
            row = metadata.get(memory_id)
            if row is None:
                # Deleted between the index read and this lookup.
                continue
            external_key, title, kind, occurred_at, source_name = row
            hits.append(
                MemoryHit(
                    memory_id=memory_id,
                    external_key=external_key,
                    source_name=source_name,
                    title=title,
                    kind=MemoryKind(kind),
                    occurred_at=occurred_at,
                    # Max, not mean. A long document with one perfectly
                    # relevant paragraph should outrank a short one that is
                    # vaguely on-topic throughout; mean would penalise the long
                    # document for the parts that are not about the query.
                    score=max(chunk.score for chunk in matched),
                    # Ordinal order, because the reader wants them in document
                    # order rather than score order once they open the item.
                    matched_chunks=sorted(matched, key=lambda chunk: chunk.ordinal),
                )
            )

        hits.sort(key=_ranking, reverse=True)
        return hits[:k]

    async def _memory_metadata(
        self, memory_ids: list[UUID]
    ) -> dict[UUID, tuple[str, str | None, str, object, str]]:
        stmt = (
            select(
                models.Memory.id,
                models.Memory.external_key,
                models.Memory.title,
                models.Memory.kind,
                models.Memory.occurred_at,
                models.Source.name,
            )
            .join(models.Source, models.Source.id == models.Memory.source_id)
            .where(models.Memory.id.in_(memory_ids))
        )

        async with self._sessions() as session:
            rows = await session.execute(stmt)
            return {row[0]: (row[1], row[2], row[3], row[4], row[5]) for row in rows}


def fuse(
    vector_chunks: Sequence[ScoredChunk],
    keyword_chunks: Sequence[ScoredChunk],
    *,
    signals: SignalTable | None = None,
    weights: FusionWeights | None = None,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[ScoredChunk]:
    """One ranking from four, each chunk carrying where it came from.

    Signals enter as *additional rankings*, not as multipliers on a fused score.
    A post-hoc multiplier would be a second, differently-shaped mechanism whose
    interaction with RRF nobody could reason about, and it would put recency on
    a scale that has to be reconciled with a fused score — the exact problem
    RRF exists to avoid. As rankings they go through the same 1/(k+rank) sum,
    carry their own weight, and stay individually inspectable in the breakdown.

    Weighting them below the retrievers is the whole design. At 0.3, a signal's
    top-ranked chunk contributes 0.3/61 against a retriever's 1/61, so recency
    can break a tie between two plausible answers and cannot manufacture one.

    Degrades rather than fails when any ranking is empty. A stopword-only query
    produces no keyword results, an unscored corpus produces no importance
    ranking, and an empty ranking contributes no terms to the sum — so fusion
    collapses to whatever remains, reordered by a function monotonic in rank,
    which preserves that ordering exactly. Nothing is special-cased, because a
    special case is a branch that only runs where nobody tests.
    """
    resolved = weights or FusionWeights()
    table = signals or SignalTable()

    by_id: dict[str, ScoredChunk] = {}
    vector_ranks: dict[str, int] = {}
    keyword_ranks: dict[str, int] = {}

    for rank, chunk in enumerate(vector_chunks, start=1):
        by_id.setdefault(str(chunk.chunk_id), chunk)
        vector_ranks.setdefault(str(chunk.chunk_id), rank)
    for rank, chunk in enumerate(keyword_chunks, start=1):
        # The two retrievers return the same row with different scores. Either
        # copy carries the same text and offsets, so the first one wins and the
        # scores are preserved separately below.
        by_id.setdefault(str(chunk.chunk_id), chunk)
        keyword_ranks.setdefault(str(chunk.chunk_id), rank)

    vector_scores = {str(chunk.chunk_id): chunk.score for chunk in vector_chunks}
    keyword_scores = {str(chunk.chunk_id): chunk.score for chunk in keyword_chunks}

    # A signal contributes nothing at weight 0, so the ranking is not even
    # built. That is what makes "weights at zero reproduces M2.2 exactly" true
    # by construction rather than by arithmetic that happens to cancel.
    recency_ranking = (
        table.ranking(table.recency) if resolved.recency and table.recency else []
    )
    importance_ranking = (
        table.ranking(table.importance)
        if resolved.importance and table.importance
        else []
    )
    recency_ranks = {key: rank for rank, key in enumerate(recency_ranking, start=1)}
    importance_ranks = {
        key: rank for rank, key in enumerate(importance_ranking, start=1)
    }

    fused = reciprocal_rank_fusion(
        [
            [str(chunk.chunk_id) for chunk in vector_chunks],
            [str(chunk.chunk_id) for chunk in keyword_chunks],
            recency_ranking,
            importance_ranking,
        ],
        k=rrf_k,
        weights=[
            resolved.vector,
            resolved.keyword,
            resolved.recency,
            resolved.importance,
        ],
    )

    return [
        replace(
            by_id[identifier],
            # The fused score *is* the score from here on. Everything downstream
            # ranks on this field, and a list mixing cosine similarities with
            # RRF scores would sort into nonsense.
            score=score,
            breakdown=ScoreBreakdown(
                fused=score,
                vector_rank=vector_ranks.get(identifier),
                keyword_rank=keyword_ranks.get(identifier),
                vector_score=vector_scores.get(identifier),
                keyword_score=keyword_scores.get(identifier),
                recency_rank=recency_ranks.get(identifier),
                importance_rank=importance_ranks.get(identifier),
                recency_score=table.recency.get(identifier),
                importance_score=table.importance.get(identifier),
            ),
        )
        for identifier, score in fused
    ]


def _attribute_single(
    chunks: Sequence[ScoredChunk], mode: SearchMode
) -> list[ScoredChunk]:
    """Fill in the breakdown for a single-retriever search.

    Populated in every mode rather than only under hybrid, so a consumer never
    has to ask which mode produced a result before it can read the provenance.
    `fused` is the retriever's own score here, which is honest: it names what
    the ranking was actually made from.
    """
    keyword = mode is SearchMode.KEYWORD
    return [
        replace(
            chunk,
            breakdown=ScoreBreakdown(
                fused=chunk.score,
                vector_rank=None if keyword else rank,
                keyword_rank=rank if keyword else None,
                vector_score=None if keyword else chunk.score,
                keyword_score=chunk.score if keyword else None,
            ),
        )
        for rank, chunk in enumerate(chunks, start=1)
    ]


def _ranking(hit: MemoryHit) -> tuple[float, float]:
    """Best chunk first, then mean as the tie-break.

    Two documents whose best chunk scores identically are separated by how much
    of the rest is also relevant — which is the one place a mean is the right
    summary.
    """
    return (hit.score, mean(chunk.score for chunk in hit.matched_chunks))


def _elapsed_ms(started: float) -> int:
    return int((time.monotonic() - started) * 1000)
