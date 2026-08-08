"""Search with the real model.

The fake embedder proves the plumbing: operators, filters, grouping, plans. It
cannot prove that the results mean anything, because a hash-based vector has no
notion of meaning. These tests are the only ones that do.
"""

from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from memoryos.adapters.blobs.filesystem import FilesystemBlobStore
from memoryos.adapters.connectors.filesystem import FilesystemConnector
from memoryos.adapters.db import models
from memoryos.adapters.db.embedding_cache import PostgresEmbeddingCache
from memoryos.adapters.db.repositories import SqlAlchemySourceRepository
from memoryos.adapters.db.vector_store import PgVectorStore
from memoryos.adapters.embedding.sentence_transformers import SentenceTransformerEmbedder
from memoryos.adapters.parsers.registry import build_default_registry as build_parsers
from memoryos.application.embed import EmbedMemory
from memoryos.application.evaluation import measure_recall
from memoryos.application.normalize import NormalizeMemory
from memoryos.application.search import SearchMemories
from memoryos.application.sync import SyncSource
from memoryos.domain.entities import Source
from memoryos.domain.ids import new_id
from memoryos.domain.jobs import JobType
from memoryos.domain.values import SourceKind

pytestmark = [pytest.mark.slow, pytest.mark.integration]

# Three about one topic, three about another. Deliberately sharing no
# distinctive vocabulary across the groups, so a keyword matcher would do as
# well — the point is that the *embeddings* separate them.
QUEUE_DOCS = {
    "queue-claim.md": (
        "# Claiming work\n\n"
        "A worker takes the next available task from the queue and begins running it. "
        "Claiming marks the row so that no other worker picks up the same task. "
        "The claim is held for a fixed period and renewed while the work continues. "
    ) * 3,
    "queue-lease.md": (
        "# Leases\n\n"
        "Each claimed task carries a lease with an expiry. If the process running it "
        "dies, the lease lapses and a sweeper returns the task to the pending pool. "
        "Renewing the lease is how a long-running task keeps its hold on the work. "
    ) * 3,
    "queue-retry.md": (
        "# Retries and backoff\n\n"
        "A failed task is scheduled to run again after a delay that grows with each "
        "attempt. Jitter spreads the retries apart so that many tasks failing together "
        "do not all come back at the same instant. "
    ) * 3,
}

BAKING_DOCS = {
    "bread-sourdough.md": (
        "# Sourdough\n\n"
        "A wild yeast starter is fed flour and water until it doubles reliably. "
        "The dough is folded gently and given a long cold rest in the refrigerator, "
        "which develops a deeper sour flavour in the finished loaf. "
    ) * 3,
    "bread-crust.md": (
        "# Crust\n\n"
        "Steam in the first minutes of baking keeps the surface of the loaf soft so it "
        "can expand. Once the steam is released the crust darkens and crisps. "
        "A very hot oven and a heavy covered pot both help. "
    ) * 3,
    "bread-flour.md": (
        "# Flour\n\n"
        "Bread flour has more protein than pastry flour, which gives the dough enough "
        "gluten to hold the gas produced during fermentation. Whole grain flours "
        "absorb more water and ferment faster. "
    ) * 3,
}


@pytest.fixture
async def searcher(
    tmp_path: Path, sessions: async_sessionmaker[AsyncSession]
) -> SearchMemories:
    root = tmp_path / "corpus"
    root.mkdir()
    for name, body in {**QUEUE_DOCS, **BAKING_DOCS}.items():
        (root / name).write_text(body)

    source = Source(
        id=new_id(), kind=SourceKind.FILESYSTEM, name="corpus", config={"root": str(root)}
    )
    async with sessions.begin() as session:
        await SqlAlchemySourceRepository(session).add(source)

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    embedder = SentenceTransformerEmbedder()
    await SyncSource(sessions, FilesystemConnector(blobs), blobs)(source.id, full=True)

    normalize = NormalizeMemory(sessions, blobs, build_parsers())
    embed = EmbedMemory(sessions, embedder, PostgresEmbeddingCache(sessions))
    for job_type, handler in (
        (JobType.NORMALIZE_MEMORY, normalize),
        (JobType.EMBED_MEMORY, embed),
    ):
        async with sessions() as session:
            targets = [
                UUID(row[0]["memory_id"])
                for row in await session.execute(
                    select(models.Job.payload).where(models.Job.job_type == job_type.value)
                )
            ]
        for memory_id in targets:
            await handler(memory_id)

    return SearchMemories(sessions, embedder, PgVectorStore(sessions, embedder))


async def test_a_topical_query_returns_that_topic_above_the_others(
    searcher: SearchMemories,
) -> None:
    """The assertion a fake embedder cannot make.

    Six documents, two topics, no shared distinctive vocabulary. If the top
    three for a queue question are not the three queue documents, the vectors
    are not carrying meaning — and every structural test in this milestone
    would still be green.
    """
    result = await searcher("how does a worker take a task and hold onto it", k=6)

    assert len(result.hits) == 6
    top_three = {hit.external_key for hit in result.hits[:3]}
    assert top_three == set(QUEUE_DOCS), [
        (hit.external_key, round(hit.score, 3)) for hit in result.hits
    ]

    # And a clear margin, not a coin flip.
    assert result.hits[2].score > result.hits[3].score + 0.05


async def test_the_other_topic_separates_the_same_way(
    searcher: SearchMemories,
) -> None:
    result = await searcher("how do I get a crisp crust on a loaf of bread", k=6)

    top_three = {hit.external_key for hit in result.hits[:3]}
    assert top_three == set(BAKING_DOCS), [
        (hit.external_key, round(hit.score, 3)) for hit in result.hits
    ]


async def test_search_returns_the_chunks_that_matched(
    searcher: SearchMemories,
) -> None:
    result = await searcher("leases expire when a worker dies", k=3)

    top = result.hits[0]
    assert top.matched_chunks
    # Chunk-level evidence for a memory-level answer.
    assert any("lease" in chunk.text.lower() for chunk in top.matched_chunks)


async def test_recall_is_high_at_the_default_ef_search(
    searcher: SearchMemories, sessions: async_sessionmaker[AsyncSession]
) -> None:
    embedder = SentenceTransformerEmbedder()
    rows = await measure_recall(
        sessions,
        embedder,
        PgVectorStore(sessions, embedder),
        queries=6,
        k=5,
        ef_search_values=[100],
    )

    (row,) = rows
    assert row.mean_recall >= 0.9, row
    # A chunk used as its own query must come back first, or something upstream
    # of the index is wrong.
    assert row.self_retrieval == pytest.approx(1.0)
