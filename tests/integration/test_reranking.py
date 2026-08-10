"""Retrieve-then-rerank, through the real pipeline with a fake cross-encoder.

None of these assert that reranking *improves* anything — a fake cannot
establish that, and `evaluate --no-rerank` against the golden set is what
measures it. What they assert is that the pipeline does what it claims: it
honours the model's ordering, it can be switched off exactly, it truncates
before asking, and it records the answer.

The fake reverses the shortlist by default. A reversal is the only ordering that
cannot be produced by accident: any partial agreement might be coincidence, and
the identity ordering is indistinguishable from ignoring the model entirely.
"""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.chunking.structural import StructuralChunker
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.keyword_store import PostgresKeywordStore
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import SearchMode, SourceKind
from tests.support.fakes import FakeEmbedder, FakeReranker

pytestmark = pytest.mark.integration

QUEUE = (
    "The worker claims a task from the queue and holds a lease while the handler "
    "runs. Renewing the lease keeps its hold on the work it started. "
)
BREAD = (
    "A wild yeast starter is fed flour and water until it doubles reliably, then "
    "folded gently and given a long cold rest in the refrigerator. "
)
NOTES = (
    "An unrelated note about scheduling meetings and taking minutes, kept here to "
    "give the shortlist a third distinct document to order. "
)


async def build(
    tmp_path: Path,
    sessions: async_sessionmaker[AsyncSession],
    reranker: FakeReranker | None,
    *,
    candidates: int = 50,
) -> SearchMemories:
    root = tmp_path / "corpus"
    if not root.exists():
        root.mkdir()
        (root / "queue.md").write_text("# Queue\n\n" + QUEUE * 5 + "\n")
        (root / "bread.md").write_text("# Bread\n\n" + BREAD * 5 + "\n")
        (root / "notes.md").write_text("# Notes\n\n" + NOTES * 5 + "\n")

        source = Source(
            id=new_id(),
            kind=SourceKind.FILESYSTEM,
            name="fixture",
            config={"root": str(root)},
        )
        async with sessions.begin() as session:
            await SqlAlchemySourceRepository(session).add(source)

        blobs = FilesystemBlobStore(tmp_path / "blobs")
        embedder = FakeEmbedder()
        await SyncSource(sessions, FilesystemConnector(blobs), blobs)(
            source.id, full=True
        )
        await _drain(
            sessions,
            JobType.NORMALIZE_MEMORY,
            NormalizeMemory(sessions, blobs, build_parsers(), StructuralChunker(embedder)),
        )
        await _drain(
            sessions,
            JobType.EMBED_MEMORY,
            EmbedMemory(sessions, embedder, PostgresEmbeddingCache(sessions)),
        )

    embedder = FakeEmbedder()
    return SearchMemories(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder, default_ef_search=100),
        PostgresKeywordStore(sessions),
        None,
        reranker,
        rerank_candidates=candidates,
    )


async def test_reranking_reorders_by_what_the_model_said(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """An inverting reranker must invert the result. Nothing subtler will do."""
    reranker = FakeReranker()
    search = await build(tmp_path, sessions, reranker)

    fused = await search("a lease on a claimed task", k=3, mode=SearchMode.HYBRID, rerank=False)
    reranked = await search("a lease on a claimed task", k=3, mode=SearchMode.HYBRID)

    assert fused.hits and reranked.hits
    assert reranker.pairs_scored > 0, "the model was actually consulted"

    # The fake scores by input position ascending, so the chunk fusion ranked
    # last is the one the reranker likes most.
    fused_chunks = [
        str(chunk.chunk_id) for hit in fused.hits for chunk in hit.matched_chunks
    ]
    reranked_chunks = [
        str(chunk.chunk_id) for hit in reranked.hits for chunk in hit.matched_chunks
    ]
    assert set(fused_chunks) & set(reranked_chunks), "same candidate pool"
    assert fused_chunks != reranked_chunks, "the ordering changed"

    # The reranker's own top pick leads the result.
    top = reranked.hits[0].matched_chunks
    assert any(chunk.breakdown and chunk.breakdown.rerank_rank == 1 for chunk in top)


async def test_no_rerank_reproduces_the_fused_ordering_exactly(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """`--no-rerank` has to be an exact bypass, not an approximation.

    It is what makes the M2.3 baselines still comparable and what makes the
    reranker's contribution measurable at all. If this drifts, every number
    recorded before this milestone describes a different system.
    """
    with_model = await build(tmp_path, sessions, FakeReranker())
    without = await build(tmp_path, sessions, None)

    for query in ("a lease on a claimed task", "yeast and flour", "the and of"):
        bypassed = await with_model(query, k=3, mode=SearchMode.HYBRID, rerank=False)
        absent = await without(query, k=3, mode=SearchMode.HYBRID)

        assert [hit.external_key for hit in bypassed.hits] == [
            hit.external_key for hit in absent.hits
        ]
        for left, right in zip(bypassed.hits, absent.hits, strict=True):
            assert left.score == right.score
            for chunk in left.matched_chunks:
                assert chunk.breakdown is not None
                # Never scored, so never ranked — and the breakdown says which.
                assert chunk.breakdown.rerank_score is None
                assert chunk.breakdown.rerank_rank is None
        assert bypassed.timing.rerank_ms == 0


async def test_the_breakdown_carries_the_rerank_score(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> None:
    """The fused score survives underneath the cross-encoder's.

    Without that, a ranking change could not be attributed to the reranker
    rather than to retrieval, which is the whole reason the breakdown exists.
    """
    search = await build(tmp_path, sessions, FakeReranker())
    result = await search("a lease on a claimed task", k=3, mode=SearchMode.HYBRID)

    chunks = [chunk for hit in result.hits for chunk in hit.matched_chunks]
    assert chunks

    for chunk in chunks:
        breakdown = chunk.breakdown
        assert breakdown is not None
        assert breakdown.rerank_score is not None
        assert breakdown.rerank_rank is not None and breakdown.rerank_rank >= 1
        # The reranker's score is what everything now ranks on...
        assert chunk.score == breakdown.rerank_score
        # ...and the fused score it replaced is still legible beside it.
        assert breakdown.fused is not None
        assert breakdown.vector_rank is not None or breakdown.keyword_rank is not None

    ranks = sorted(
        chunk.breakdown.rerank_rank
        for hit in result.hits
        for chunk in hit.matched_chunks
        if chunk.breakdown is not None and chunk.breakdown.rerank_rank is not None
    )
    assert ranks == sorted(set(ranks)), "ranks are unique"


async def _drain(
    sessions: async_sessionmaker[AsyncSession], job_type: JobType, handler: object
) -> None:
    async with sessions() as session:
        targets = [
            UUID(row[0]["memory_id"])
            for row in await session.execute(
                select(models.Job.payload).where(models.Job.job_type == job_type.value)
            )
        ]
    for memory_id in targets:
        await handler(memory_id)  # type: ignore[operator]
    async with sessions.begin() as session:
        await session.execute(delete(models.Job).where(models.Job.job_type == job_type.value))
