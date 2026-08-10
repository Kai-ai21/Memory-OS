"""Hybrid search through the real path, against both real indexes.

Two properties, and both are about what happens at the seams rather than about
ranking quality — quality is what `evaluate` measures, over a golden set, and a
test that asserted "the right document comes first" against a fake embedder
would be asserting a hash.

The seams: a retriever returning nothing must degrade rather than erase the
other one's results, and every chunk must be able to say which retrievers found
it and where.
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
from tests.support.fakes import FakeEmbedder

pytestmark = pytest.mark.integration

CLAIM = (
    "The claim query takes the oldest pending job and marks it running. "
    "FOR UPDATE SKIP LOCKED on the inner select is the clause that makes two "
    "workers claim different rows instead of queueing behind each other. "
)
LEASE = (
    "Renewing the lease is how a long running handler keeps its hold on the "
    "work it started, so a sweeper does not reclaim a job that is progressing. "
)
BREAD = (
    "A wild yeast starter is fed flour and water until it doubles reliably, "
    "then folded gently and given a long cold rest in the refrigerator. "
)


@pytest.fixture
async def search(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> SearchMemories:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "queue.md").write_text("# Queue\n\n" + CLAIM * 4 + "\n")
    (root / "lease.md").write_text("# Lease\n\n" + LEASE * 4 + "\n")
    (root / "bread.md").write_text("# Bread\n\n" + BREAD * 4 + "\n")

    source = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name="fixture", config={"root": str(root)}
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    embedder = FakeEmbedder()
    await SyncSource(sessions, FilesystemConnector(blobs), blobs)(source.id, full=True)
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

    return SearchMemories(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder, default_ef_search=100),
        PostgresKeywordStore(sessions),
    )


async def test_an_empty_keyword_side_degrades_to_pure_vector(
    search: SearchMemories,
) -> None:
    """A stopword query is the ordinary way one retriever returns nothing.

    RRF must treat an empty ranking as contributing no terms rather than as
    evidence against everything the other retriever found. And because
    `1/(k+rank)` is monotonically decreasing in rank, fusing one ranking with
    nothing reproduces that ranking exactly — the degradation is to *pure
    vector*, not merely to *something*.
    """
    stopwords = await search("the and of", k=3, mode=SearchMode.HYBRID)
    vector_only = await search("the and of", k=3, mode=SearchMode.VECTOR)

    assert stopwords.hits, "the vector half still had an answer"
    assert [hit.external_key for hit in stopwords.hits] == [
        hit.external_key for hit in vector_only.hits
    ]

    for hit in stopwords.hits:
        for chunk in hit.matched_chunks:
            assert chunk.breakdown is not None
            # Nothing was found lexically, and the breakdown says so rather than
            # reporting a rank of zero or a score of 0.0.
            assert chunk.breakdown.keyword_rank is None
            assert chunk.breakdown.keyword_score is None
            assert chunk.breakdown.vector_rank is not None


async def test_a_chunk_found_by_both_carries_both_ranks(
    search: SearchMemories,
) -> None:
    """The explainability guardrail, asserted rather than assumed.

    The fused score is the only number a reader sees, and on its own it explains
    nothing: 0.0325 means nothing until it can be read as "rank 1 here, rank 2
    there". Every chunk in every mode carries that, and `fused` always names
    the number the ranking was actually made from.
    """
    result = await search("SKIP LOCKED workers claim rows", k=3, mode=SearchMode.HYBRID)

    assert result.mode is SearchMode.HYBRID
    chunks = [chunk for hit in result.hits for chunk in hit.matched_chunks]
    assert chunks

    both = [
        chunk
        for chunk in chunks
        if chunk.breakdown is not None
        and chunk.breakdown.vector_rank is not None
        and chunk.breakdown.keyword_rank is not None
    ]
    assert both, "the queue text is reachable by either retriever, so some chunk is in both"

    for chunk in both:
        breakdown = chunk.breakdown
        assert breakdown is not None
        assert breakdown.vector_rank is not None and breakdown.vector_rank >= 1
        assert breakdown.keyword_rank is not None and breakdown.keyword_rank >= 1
        assert breakdown.vector_score is not None
        assert breakdown.keyword_score is not None
        # The fused score replaced the retriever's own, and the raw ones survive
        # underneath it — which is what makes a regression attributable to a half.
        assert chunk.score == breakdown.fused
        assert breakdown.fused != breakdown.vector_score

    # And a single-retriever search fills the same field, so a consumer never
    # has to know the mode before it can read provenance.
    keyword = await search("SKIP LOCKED", k=3, mode=SearchMode.KEYWORD)
    for hit in keyword.hits:
        for chunk in hit.matched_chunks:
            assert chunk.breakdown is not None
            assert chunk.breakdown.keyword_rank is not None
            assert chunk.breakdown.vector_rank is None
            assert chunk.breakdown.fused == chunk.score


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
